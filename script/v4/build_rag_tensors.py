import sys
from pathlib import Path
import numpy as np
import pandas as pd
import h5py
import faiss
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor, CLIPModel
from tqdm.auto import tqdm
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.shared.config import (
    BATCH_SIZE, TRAIN_DF_PATH, VAL_DF_PATH, TEST_DF_PATH,
    KB_MODEL_ID, IMAGES_DIR,
    IMAGE_KB_FAISS_INDEX_PATH, IMAGE_KB_METADATA_PATH,
    TRAIN_VISUAL_FEATURES_PATH, VAL_VISUAL_FEATURES_PATH, TEST_VISUAL_FEATURES_PATH,
    TRAIN_RAG_TENSORS_PATH, VAL_RAG_TENSORS_PATH, TEST_RAG_TENSORS_PATH,
    TOP_K_RAG_IMAGES,
    NUM_WORKERS,
)
from src.shared.utils import extract_clip_features
from src.shared.dataset import RawImageDataset
from src.shared.encoder import CLIPImageEmbeddingEncoder


def load_h5_to_ram(h5_path: Path):
    print(f"  Loading {h5_path.name} into RAM...", flush=True)
    with h5py.File(h5_path, "r") as f:
        imgids = np.array(f["imgids"])             
        features = np.array(f["features"])[:, 1:, :]  
    imgid2idx = {int(iid): i for i, iid in enumerate(imgids)}
    return features, imgid2idx


def encode_image_features(df, vision_encoder, target_dtype, processor, device) -> tuple[list[int], np.ndarray]:
    df_unique = df.drop_duplicates(subset=["imgid"]).reset_index(drop=True)
    dataset = RawImageDataset(df_unique, IMAGES_DIR, processor=processor)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE*2, shuffle=False, num_workers=NUM_WORKERS)
    all_img_features, all_imgids = [], []
    with torch.no_grad():
        for pixels, batch_imgids in tqdm(loader, desc="  Encode global embedding", leave=False):
            pixels = pixels.to(device, dtype=target_dtype)
            vecs = vision_encoder(pixel_values=pixels)
            # CRITICAL FIX: Cast to float32 BEFORE norm to avoid NaN!
            vecs = vecs.to(torch.float32)
            vecs = F.normalize(vecs, p=2, dim=-1)
            all_img_features.append(vecs.cpu().numpy())
            all_imgids.extend(batch_imgids.tolist())
    return all_imgids, np.concatenate(all_img_features, axis=0)

def build_rag_tensors(
    split_name: str, query_imgids: list[int], query_image_features: np.ndarray,
    faiss_index, kb_imgids: list[int], query_patch_features: np.ndarray, query_imgid2idx: dict,
    kb_patch_features: np.ndarray, kb_imgid2idx: dict, output_path: Path, device: str
):
    num_queries = len(query_imgids)
    num_patches = query_patch_features.shape[1]
    d_model = query_patch_features.shape[2]
    
    print(f"[{split_name}] Building RAG Tensors → {output_path.name}", flush=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    WRITE_BATCH = 512  

    with h5py.File(output_path, "w") as f_out:
        f_out.create_dataset("imgids", data=np.array(query_imgids, dtype=np.int64))
        ds_retrieved = f_out.create_dataset(
            "retrieved_imgids", shape=(num_queries, TOP_K_RAG_IMAGES), dtype=np.int64, chunks=(1, TOP_K_RAG_IMAGES)
        )
        ds = f_out.create_dataset(
            "features", shape=(num_queries, num_patches, d_model), dtype="float16", chunks=(1, num_patches, d_model)
        )

        for start in tqdm(range(0, num_queries, WRITE_BATCH), desc=f"  [{split_name}] RAG"):
            end = min(start + WRITE_BATCH, num_queries)
            batch_size = end - start
            
            # On-the-fly FAISS search for this chunk
            batch_query_image_features = query_image_features[start:end]
            _distances, indices = faiss_index.search(batch_query_image_features, TOP_K_RAG_IMAGES + 1)
            
            # Prepare CPU numpy arrays to gather data before sending to GPU
            q_batch_cpu = np.empty((batch_size, num_patches, d_model), dtype=np.float16)
            k_batch_cpu = np.empty((batch_size, TOP_K_RAG_IMAGES * num_patches, d_model), dtype=np.float16)
            retrieved_imgids_batch = np.empty((batch_size, TOP_K_RAG_IMAGES), dtype=np.int64)

            for i, global_idx in enumerate(range(start, end)):
                query_imgid = query_imgids[global_idx]
                q_batch_cpu[i] = query_patch_features[query_imgid2idx[query_imgid]]

                valid_retrieved_imgids = []
                for j, kb_row_idx in enumerate(indices[i]):
                    if kb_row_idx == -1: continue
                    retrieved_imgid = kb_imgids[kb_row_idx]
                    if retrieved_imgid != query_imgid:
                        valid_retrieved_imgids.append(retrieved_imgid)
                    if len(valid_retrieved_imgids) == TOP_K_RAG_IMAGES:
                        break
                    
                if len(valid_retrieved_imgids) < TOP_K_RAG_IMAGES:
                    raise RuntimeError(f"Lỗi Logic: FAISS chỉ tìm được {len(valid_retrieved_imgids)}/{TOP_K_RAG_IMAGES} ảnh cho query_imgid={query_imgid}. Kiểm tra lại kích thước tập Train!")

                retrieved_imgids_batch[i] = valid_retrieved_imgids

                k_features = [kb_patch_features[kb_imgid2idx[imgid]] for imgid in valid_retrieved_imgids]
                k_batch_cpu[i] = np.concatenate(k_features, axis=0)

            COMPUTE_BATCH = BATCH_SIZE*2
            final_rag_cpu = np.empty((batch_size, num_patches, d_model), dtype=np.float16)

            for c_start in range(0, batch_size, COMPUTE_BATCH):
                c_end = min(c_start + COMPUTE_BATCH, batch_size)
                
                q_batch = torch.tensor(q_batch_cpu[c_start:c_end], device=device, dtype=torch.float32)
                k_batch = torch.tensor(k_batch_cpu[c_start:c_end], device=device, dtype=torch.float32)

                # Chuẩn hoá L2 biến độ dài vector thành 1, giúp kết quả của tích vô hướng = cosine similarity. 
                q_norm = F.normalize(q_batch, p=2, dim=-1) 
                k_norm = F.normalize(k_batch, p=2, dim=-1) 
                
                # Nhân hai ma trận thực chất là cách tính đồng thời (song song) tất cả các tích vô hướng 
                # giữa các vector hàng của ma trận này với các vector cột của ma trận kia.
                # vector_u . vector_v = |u|.|v|.cosine(góc kẹp giữa)
                similarity = torch.bmm(q_norm, k_norm.transpose(1, 2)) # shape: [COMPUTE_BATCH, num_patches, TOP_K * num_patches] (e.g. [64, 49, 196])
                best_patch_indices = torch.argmax(similarity, dim=-1, keepdim=True) # shape: [COMPUTE_BATCH, num_patches, 1] (e.g. [64, 49, 1])

                best_patch_indices_expanded = best_patch_indices.expand(-1, -1, d_model) # shape: [COMPUTE_BATCH, num_patches, d_model] (e.g. [64, 49, 768])
                # Giải thích gather(dim=1):
                # final_rag_tensor[i][j][k] = k_batch[i][ best_patch_indices_expanded[i][j][k] ][k]
                final_rag_tensor = torch.gather(k_batch, 1, best_patch_indices_expanded) # shape: [COMPUTE_BATCH, num_patches, d_model] (e.g. [64, 49, 768]) 
                
                final_rag_cpu[c_start:c_end] = final_rag_tensor.to(torch.float16).cpu().numpy()

            # Save
            ds[start:end] = final_rag_cpu
            ds_retrieved[start:end] = retrieved_imgids_batch

    print(f"[{split_name}] ✓ Saved {num_queries:,} RAG Tensors → {output_path}", flush=True)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print(f"\nLoading CLIP model ({KB_MODEL_ID})...")
    target_dtype = torch.float16 if device == "cuda" else torch.float32
    vision_encoder = CLIPImageEmbeddingEncoder(torch_dtype=target_dtype).eval().to(device)
    processor = CLIPProcessor.from_pretrained(KB_MODEL_ID)

    if device == "cuda" and torch.cuda.device_count() > 1:
        print(f"Bật chế độ Multi-GPU DataParallel với {torch.cuda.device_count()} GPUs!")
        vision_encoder = torch.nn.DataParallel(vision_encoder)

    print("Loading FAISS index...")
    faiss_index = faiss.read_index(str(IMAGE_KB_FAISS_INDEX_PATH))
    kb_metadata_df = pd.read_parquet(IMAGE_KB_METADATA_PATH)
    kb_imgids = kb_metadata_df["imgid"].tolist()

    print("\n--- Loading Knowledge Base (TRAIN) ---")
    train_patch_features, train_imgid2idx = load_h5_to_ram(TRAIN_VISUAL_FEATURES_PATH)
    splits = [
        ("TRAIN", TRAIN_VISUAL_FEATURES_PATH, TRAIN_DF_PATH, TRAIN_RAG_TENSORS_PATH),
        ("VAL", VAL_VISUAL_FEATURES_PATH, VAL_DF_PATH, VAL_RAG_TENSORS_PATH),
        ("TEST", TEST_VISUAL_FEATURES_PATH, TEST_DF_PATH, TEST_RAG_TENSORS_PATH),
    ]
    for name, vis_path, df_path, rag_path in splits:
        print(f"\n--- Processing {name} ---")
        
        if name == "TRAIN":
            query_patch_features, query_imgid2idx = train_patch_features, train_imgid2idx
        else:
            query_patch_features, query_imgid2idx = load_h5_to_ram(vis_path)
            
        df = pd.read_parquet(df_path)
        imgids, img_features = encode_image_features(df, vision_encoder, target_dtype, processor, device)
        
        build_rag_tensors(
            name, imgids, img_features, faiss_index, kb_imgids,
            query_patch_features, query_imgid2idx, train_patch_features, train_imgid2idx,
            rag_path, device
        )
        
        if name != "TRAIN":
            del query_patch_features, query_imgid2idx


if __name__ == "__main__":
    main()
