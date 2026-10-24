"""FACTUAL graph normalization and deterministic phrase realization.

Entity IDs and v:/pv: markers guide normalization but never appear in retrieval
phrases. Explicit quantities are omitted; caption tense is not inferred.
"""

import re

import torch
from factual_scene_graph.parser.scene_graph_parser import SceneGraphParser
from factual_scene_graph.utils import clean_graph_string, space_out_symbols_in_graph
from simplenlg.features import Feature, Form
from simplenlg.framework import LexicalCategory, NLGFactory
from simplenlg.lexicon import Lexicon
from simplenlg.realiser.english import Realiser
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


ENTITY_SUFFIX = re.compile(r":\d+$")


def object_label(head: str) -> str:
    """Remove only FACTUAL's terminal entity index, preserving compound names."""
    return ENTITY_SUFFIX.sub("", head.strip())


def predicate_parts(predicate: str) -> tuple[str, str]:
    if predicate.startswith("pv:"):
        return predicate[3:].strip(), "passive"
    if predicate.startswith("v:"):
        return predicate[2:].strip(), "active"
    return predicate.strip(), "unspecified"


class CheckedParser(SceneGraphParser):
    """Use FACTUAL's graph converter with checked, untruncated generation"""
    def __init__(self, model_id: str, device: str, revision=None):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_id, revision=revision
        ).to(device).eval()
        self.lemmatize = self.lowercase = False
        self.parser_type = "default"

    def graph_string_to_object(self, text):
        """Convert FACTUAL tuples directly and robustly into entity/relation dicts.

        Handles 1-tuples (entities), 2-tuples (attributes), 3-tuples (triplets/attributes),
        and n-tuples where attribute values or object names contain commas (e.g. sign
        inscriptions, list of items, or geographical entities).
        """
        graph = {"entities": [], "relations": []}
        entity_map = {}

        fact_texts = re.findall(r"\(([^()]*)\)", text)
        for fact_text in fact_texts:
            parts = [p.strip() for p in fact_text.split(",") if p.strip()]
            if not parts:
                continue
            if len(parts) == 1:
                self._get_or_create_entity_index(parts[0], graph, entity_map)
            elif len(parts) == 2:
                sub_idx = self._get_or_create_entity_index(parts[0], graph, entity_map)
                graph["entities"][sub_idx]["attributes"].add(parts[1])
            else:
                subject = parts[0]
                relation = parts[1]
                obj = ", ".join(parts[2:])
                sub_idx = self._get_or_create_entity_index(subject, graph, entity_map)
                if relation == "is":
                    if obj.isdigit():
                        graph["entities"][sub_idx]["quantity"] = obj
                    else:
                        graph["entities"][sub_idx]["attributes"].add(obj)
                else:
                    obj_idx = self._get_or_create_entity_index(obj, graph, entity_map)
                    graph["relations"].append({
                        "subject": sub_idx,
                        "relation": relation,
                        "object": obj_idx,
                    })

        return graph

    def _parse_batch(self, descriptions, max_input_len, max_output_len,
                     beam_size, filter_factual_chars, batch_size):
        if filter_factual_chars:
            raise ValueError("Keep FACTUAL ID/voice markers for linearization.")

        graph_texts = []
        for batch_start in range(0, len(descriptions), batch_size):
            caption_batch = descriptions[batch_start:batch_start + batch_size]
            prompts = ["Generate Scene Graph: " + caption.strip()
                       for caption in caption_batch]

            # Do not truncate captions silently; reject a batch that exceeds the configured limit.
            tokenized = self.tokenizer(
                prompts, padding=True, truncation=False, return_tensors="pt"
            )
            input_length = tokenized["input_ids"].shape[1]
            if input_length > max_input_len:
                raise ValueError(
                    "Caption exceeds max_input_len; adjust PARSING_OPTIONS in build_i2t_kb.py."
                )

            with torch.inference_mode():
                generated_ids = self.model.generate(
                    tokenized["input_ids"].to(self.device),
                    attention_mask=tokenized["attention_mask"].to(self.device),
                    decoder_start_token_id=self.tokenizer.pad_token_id,
                    num_beams=beam_size, max_length=max_output_len,
                    early_stopping=beam_size > 1,
                    num_return_sequences=1, use_cache=True,
                )

            decoded_graphs = self.tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            for token_ids, raw_graph in zip(generated_ids, decoded_graphs):
                if self.tokenizer.eos_token_id not in token_ids.tolist():
                    raise ValueError(
                        "Graph reached the output limit; adjust PARSING_OPTIONS in build_i2t_kb.py."
                    )

                raw_graph = raw_graph.replace("Generate Scene Graph:", "").strip()
                fact_texts = re.findall(r"\(([^()]*)\)", raw_graph)
                text_outside_facts = re.sub(r"\([^()]*\)", "", raw_graph)
                text_outside_facts = text_outside_facts.strip(" ,\n\t")

                malformed = not fact_texts or bool(text_outside_facts)
                for fact_text in fact_texts:
                    raw_parts = fact_text.split(",")
                    parts = [p.strip() for p in raw_parts if p.strip()]
                    has_wrong_number_of_parts = len(parts) < 1
                    has_empty_part = any(not p.strip() for p in raw_parts)
                    if has_wrong_number_of_parts or has_empty_part:
                        malformed = True
                        break

                if malformed:
                    raise ValueError(f"Malformed FACTUAL output: {raw_graph!r}")

                cleaned_graph = clean_graph_string(
                    space_out_symbols_in_graph(raw_graph)
                )
                graph_texts.append(cleaned_graph)

        return graph_texts


class GraphLinearizer:
    """Build article-free phrases, preserving FACTUAL's noun forms.

    SimpleNLG realizes only verb participles. Its lexicon also distinguishes
    adjective modifiers from attributes placed after the object label.
    """

    def __init__(self):
        self.feature, self.form, self.category = Feature, Form, LexicalCategory
        self.lexicon = Lexicon.getDefaultLexicon()
        self.factory = NLGFactory(self.lexicon)
        self.realiser = Realiser(self.lexicon)

    def _text(self, phrase) -> str:
        return self.realiser.realise(phrase).getRealisation().strip()

    def _noun(self, entity, adjective=None) -> str:
        """Keep the supplied noun form without a determiner or quantity."""
        label = object_label(entity["head"])
        return f"{adjective} {label}" if adjective else label

    def _edge_phrase(self, subject, predicate, obj=None):
        predicate, voice = predicate_parts(predicate)
        if not predicate:
            raise ValueError("An empty predicate cannot be linearized.")
        if voice != "unspecified":
            negated = predicate.startswith("not ")
            if negated:
                predicate = predicate[4:]
            verb, _, rest = predicate.partition(" ")
            if not verb:
                raise ValueError("An empty verb cannot be linearized.")
            modifier = self.factory.createVerbPhrase(verb)
            modifier.setFeature(
                self.feature.FORM,
                self.form.PAST_PARTICIPLE if voice == "passive"
                else self.form.PRESENT_PARTICIPLE,
            )
            # Keep negation and the rest of the predicate outside the realiser.
            # Realising a negated participle can introduce unwanted do-support.
            predicate = " ".join(part for part in (
                "not" if negated else "", self._text(modifier), rest
            ) if part)
        parts = [self._noun(subject), predicate]
        if obj is not None:
            parts.append(self._noun(obj))
        return " ".join(parts)

    def normalize(self, graph: dict) -> dict:
        """Return objects, attributes, triplets, and their aligned phrases."""
        entities = graph["entities"]
        for entity in entities:
            if not object_label(entity["head"]):
                raise ValueError("FACTUAL returned an empty entity.")
        objects = [object_label(e["head"]) for e in entities]
        object_texts = [self._noun(e) for e in entities]
        attributes, attribute_texts, triplets, triplet_texts = [], [], [], []
        for entity, label in zip(entities, objects):
            for attribute in sorted(entity.get("attributes", [])):
                attributes.append({"object": label, "attribute": attribute})
                if attribute.startswith(("v:", "pv:")):
                    text = self._edge_phrase(entity, attribute)
                elif self.lexicon.hasWord(attribute, self.category.ADJECTIVE):
                    text = self._noun(entity, attribute)
                else:
                    # Link attribute to its entity with "is" (mirrors FACTUAL's
                    # own predicate and reads naturally for all attribute types).
                    text = f"{self._noun(entity)} is {attribute}"
                attribute_texts.append(text)
        for edge in graph["relations"]:
            subject_id, object_id = edge["subject"], edge["object"]
            if not (0 <= subject_id < len(entities) and 0 <= object_id < len(entities)):
                raise ValueError("FACTUAL returned an invalid entity reference.")
            predicate, voice = predicate_parts(edge["relation"])
            triplets.append({"subject": objects[subject_id], "predicate": predicate,
                             "object": objects[object_id], "voice": voice})
            triplet_texts.append(self._edge_phrase(
                entities[subject_id], edge["relation"], entities[object_id]
            ))
        return {
            "objects": objects, "attributes": attributes, "triplets": triplets,
            "object_texts": object_texts, "attribute_texts": attribute_texts,
            "triplet_texts": triplet_texts,
        }

