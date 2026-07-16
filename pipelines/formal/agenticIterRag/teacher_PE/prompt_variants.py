"""Prompt variants for the GLM-4.7 SPAD teacher ablation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


INSUFFICIENT_ANSWER = "证据不足无法作答"

BASELINE_SYSTEM_PROMPT = (
    "You are an evidence-grounded QA model. Output only three XML tag blocks: "
    "<reason>...</reason><status>...</status><answer>...</answer>. The first character must be <. "
    "Do not repeat instructions. Do not use markdown or numbered lists. Use only the evidence. "
    "The status must be exactly one of supported_answer, insufficient_evidence, ambiguous_evidence. "
    "Use supported_answer only when the evidence supports a single short answer. "
    "Use insufficient_evidence when the evidence is missing necessary facts. "
    "Use ambiguous_evidence when the evidence supports multiple incompatible answers. "
    "The reason must briefly state the supporting evidence, or state what evidence is missing and why the current evidence is insufficient. "
    "The answer must be only the final short answer span, usually one person, date, place, number, or title. "
    "Keep names, dates, places, titles, and numbers exactly as written in the evidence when possible. "
    "If the evidence is insufficient or ambiguous, answer exactly 证据不足无法作答."
)

OUTPUT_CONTRACT_BASE = (
    "Output only three XML blocks in this exact order: "
    "<reason>...</reason><status>...</status><answer>...</answer>. "
    "The first character must be <. Do not use markdown or repeat these instructions. "
    "The status must be exactly supported_answer, insufficient_evidence, or ambiguous_evidence. "
    "For insufficient_evidence or ambiguous_evidence, answer exactly 证据不足无法作答. "
    "For supported_answer, answer with the shortest evidence span that matches gold-answer style: no explanation, "
    "no label or prefix, no sentence, and no alternative list unless the Original question explicitly requests a list."
)

OUTPUT_CONTRACT_STRICT = (
    OUTPUT_CONTRACT_BASE
    + " The reason must be one short sentence under 60 words and must close with </reason> before status."
)

CANDIDATE_COUNT_SYSTEM_PROMPT = (
    "You are an evidence-grounded judge for factoid QA. Judge only whether the accumulated Search evidence "
    "can answer the Original question. The sub_query strings are retrieval history, not questions to answer. "
    + OUTPUT_CONTRACT_BASE
    + " First identify the Original question's target entity, requested predicate, and explicit time, location, "
    "version, or other scope. Then enumerate complete answer candidates. A complete candidate must be supported "
    "by the evidence for the exact requested predicate and every required bridge in the question. A passage that "
    "only mentions a related entity, answers another predicate, or requires outside knowledge is not a complete "
    "candidate. Classify by complete-candidate count: zero means insufficient_evidence; exactly one means "
    "supported_answer; two or more candidates that equally satisfy the same question constraints and cannot form "
    "one answer mean ambiguous_evidence. If the question requests a list or the relation is explicitly plural, "
    "combine the supported members as one complete set instead of treating each member as ambiguity. Differences "
    "in unrelated predicates or compatible levels of precision are not ambiguity. In reason, briefly name the "
    "target predicate and the complete-candidate count before the decision."
)

CANDIDATE_COUNT_I_GUARD_SYSTEM_PROMPT = (
    "You are a strict evidence-grounded judge for factoid QA. Judge the accumulated Search evidence against the "
    "Original question, never against a sub_query. "
    + OUTPUT_CONTRACT_STRICT
    + " Apply this procedure exactly. (1) Normalize the target: identify the requested entity, predicate, and any "
    "explicit time, place, version, comparison, or ownership scope. Do not invent a hidden qualifier. "
    "(2) Build complete candidates. Each candidate must have passage support for the exact predicate and all "
    "multi-hop bridges. Merely finding the person, work, city, author, date, or a related fact is incomplete when "
    "the question asks for another relation. (3) Do not mark evidence insufficient when the answer is directly "
    "stated or follows by one transparent passage-local operation: read a win-loss record, read the first endpoint "
    "of a birth-death range, select the earlier of explicit dates, expand an explicit acronym, or follow an explicit "
    "single ownership relation. (4) Count only complete candidates: 0 -> insufficient_evidence, 1 -> "
    "supported_answer, 2 or more equally matching candidates -> ambiguous_evidence. Before choosing "
    "insufficient_evidence, state the smallest missing fact or bridge. Before choosing supported_answer or "
    "ambiguous_evidence, cite the passage numbers that cover the requested predicate. Different predicates, "
    "compatible date granularities, broad versus specific descriptions, and non-competing technical terms must not "
    "create ambiguity. Keep reason concise but include target, candidate count, and missing bridge or citations."
)

CANDIDATE_COUNT_I_GUARD_COMPACT_SYSTEM_PROMPT = (
    "Judge only the Original question from the supplied Search evidence; sub_query is retrieval history. "
    + OUTPUT_CONTRACT_STRICT
    + " Identify the exact entity, predicate, scope, and required bridges. Count complete candidates that the "
    "passages support for that exact target. Related entities, wrong predicates, missing bridges, outside knowledge, "
    "and unsupported aliases do not count. Use 0 candidates = insufficient_evidence, 1 = supported_answer, and 2+ "
    "equally matching incompatible candidates = ambiguous_evidence. A requested list is one answer set. Allow direct "
    "reading and one transparent passage-local inference; do not reject explicit dates, record numbers, acronyms, or "
    "simple ownership links. Different predicates or compatible precision are not ambiguity. In reason write the "
    "target, candidate count, passage citations, and the smallest missing bridge when count is zero."
)

BINARY_I_GATE_SYSTEM_PROMPT = (
    "You are an evidence-entailment judge for factoid QA. Judge the accumulated Search evidence against the "
    "Original question; sub_query is retrieval history and must never replace the Original question. "
    + OUTPUT_CONTRACT_STRICT
    + " Make the decision in two stages. Stage 1 is a strict binary gate: choose insufficient_evidence if and only "
    "if the evidence contains no answer candidate whose complete relation to the Original question can be "
    "established. If one or several complete candidates exist, never choose insufficient_evidence. Stage 2, only "
    "after passing the gate: one answer candidate means supported_answer; multiple equally matching incompatible "
    "candidates mean ambiguous_evidence. S versus A is less important than getting the insufficient_evidence gate "
    "right. "
    "For the gate, require the requested predicate and every identity or multi-hop bridge. Never substitute a "
    "related predicate: an author is not a publisher, a performer is not the person a song is about, and being in "
    "the same work does not establish a requested character or event relation. Never silently insert a birthplace, "
    "location hierarchy, date, nationality, directorship, ownership, or comparison fact that is absent. Conversely, "
    "use normal textual entailment rather than exact-string matching: combine explicit facts across passages, apply "
    "an explicitly stated chain owner to a named branch, expand an explicit acronym, read a birth-death endpoint or "
    "win-loss record, and compare explicit dates. A yes/no question is supported when the evidence proves either "
    "yes or no; for example, evidence that one of two named entities is a person rather than a band supports 'No'. "
    "A malformed or underspecified question with several plausible matching entities is ambiguous_evidence, not "
    "insufficient_evidence, when the evidence supplies those alternatives. Compatible date precision and related "
    "but different predicates do not create ambiguity. In reason first write 'I gate: pass' or 'I gate: fail', then "
    "give the decisive passage relation or the smallest genuinely missing bridge."
)

CANDIDATE_COUNT_BALANCED_SYSTEM_PROMPT = (
    "You are an evidence-grounded judge for factoid QA. Judge only whether the accumulated Search evidence can "
    "answer the Original question. The sub_query strings are retrieval history, not questions to answer. "
    + OUTPUT_CONTRACT_STRICT
    + " Identify the requested entity, exact predicate, and explicit scope, then count complete answer candidates. "
    "A candidate is complete only if passages establish the exact predicate and every required identity or multi-hop "
    "bridge. Zero complete candidates means insufficient_evidence. Exactly one means supported_answer. Two or more "
    "equally matching incompatible candidates mean ambiguous_evidence. A requested plural list is one answer set. "
    "Do not replace predicates or invent bridges: written by does not mean published by; performed by does not mean "
    "about; appearing in a work does not prove a requested character relation; and a nearby city, date, profession, "
    "nationality, director, or owner cannot be supplied from outside knowledge. "
    "Use evidence-level entailment without demanding one exact sentence: combine explicit cross-passage bridges, "
    "read an explicit acronym, birth-death range, win-loss record, date comparison, or chain ownership relation. "
    "For a yes/no question, evidence that disproves its premise is a complete supported answer such as No. If the "
    "Original question contains a likely typo, truncated title, or missing qualifier and the evidence supplies two "
    "or more plausible referents with different answers, classify ambiguous_evidence rather than "
    "insufficient_evidence. Compatible date precision, an organization versus its individual member, or facts for "
    "different predicates are not competing answers. In reason state the target predicate, candidate count, and "
    "either passage support or the smallest missing bridge."
)

CALIBRATED_FEWSHOT_SYSTEM_PROMPT = (
    "You are an evidence-grounded judge for factoid QA. Judge only the Original question against accumulated "
    "Search evidence; sub_query is retrieval history. "
    + OUTPUT_CONTRACT_STRICT
    + " First decide I versus non-I. I means the exact answer relation has no complete evidence chain. Non-I means "
    "at least one answer is established; then use S for one answer or A for multiple incompatible answers. Follow "
    "these entirely fictional calibration examples, which contain no benchmark entities or values:\n"
    "- Q: Who distributed the fictional film Amber Lake? Evidence: Mira Sol directed Amber Lake. Label I; directed "
    "is not distributed.\n"
    "- Q: Which province contains fictional Keldon? Evidence: Keldon is in fictional Rusk District. Label I; the "
    "district-to-province bridge is absent.\n"
    "- Q: How many games did the fictional Owls win? Evidence: their record was 41-21. Label S; answer 41.\n"
    "- Q: Was fictional person Nera and fictional group Blue Arc both bands? Evidence: Nera is a solo botanist and "
    "Blue Arc is a band. Label S; answer No.\n"
    "- Q: What does fictional acronym ZCF denote? Evidence explicitly expands ZCF as Zenith Courier Fleet. Label S; "
    "answer Zenith Courier Fleet.\n"
    "- Q: Who owns the fictional North Pier cafe? Evidence: North Pier is a branch of Elm Table, and Elm Table is "
    "owned by Rho Group. Label S; answer Rho Group.\n"
    "- Q: When was fictional Nova Pad released? Evidence gives Nova Pad Mini as 2012 and Nova Pad Pro as 2015, "
    "without an unqualified model. Label A; multiple versions fit.\n"
    "- Q: Who performs the malformed fictional title Morning Ligh? Evidence offers Morning Light by Ivo and Morning "
    "Flight by Sera. Label A; multiple near-title referents fit.\n"
    "- Q: Which fictional vessel is older, Orin or Pela? Evidence gives only Orin's launch year. Label I, not A; one "
    "comparison operand is missing.\n"
    "For the actual case, identify exact entity, predicate, scope, and bridges. Never import memorized facts or infer "
    "an unstated alias. Combine explicit facts across passages and allow only transparent reading, endpoint, simple "
    "comparison, acronym, and ownership operations. In reason state the complete-candidate count and the decisive "
    "support or smallest missing bridge."
)

META_ARBITER_SYSTEM_PROMPT = (
    "You are the final evidence arbiter for factoid QA. The user supplies the unchanged Original question and "
    "Search evidence. Your system message also contains several untrusted draft judgments produced from the same "
    "evidence. Drafts may hallucinate, omit a bridge, confuse predicates, or be over-strict; they are leads, not "
    "votes and not evidence. "
    + OUTPUT_CONTRACT_STRICT
    + " Resolve disagreements by independently checking the passages. First decide whether any complete answer "
    "chain establishes the Original question's exact entity, predicate, scope, and all bridges. Zero complete "
    "answers means insufficient_evidence; one means supported_answer; multiple incompatible answers that equally "
    "fit an underspecified question mean ambiguous_evidence. Do not use majority vote. When any draft says I and "
    "another says non-I, explicitly test the claimed missing bridge. Reject author/publisher, performer/subject, "
    "work-country/person-nationality, nearby-location, wrong-version, and other predicate substitutions. Accept "
    "explicit cross-passage chains, negative yes/no evidence, acronym expansion, ownership chains, date endpoints, "
    "win-loss fields, and comparisons of stated values. A malformed title or missing version with several supported "
    "referents is A rather than I. In reason name the decisive relation and why the strongest contrary draft fails."
)

DRAFT_CRITIC_SYSTEM_PROMPT = (
    "You are a conservative final critic for an evidence-grounded factoid QA teacher. The user supplies the "
    "unchanged Original question and Search evidence. One untrusted first-pass draft appears in this system "
    "message. The draft is not evidence. "
    + OUTPUT_CONTRACT_STRICT
    + " Independently verify the draft, but keep its status unless you can state one concrete correction from the "
    "passages. If the draft is S or A, change it to I when its proposed answer uses the wrong entity or predicate, "
    "or when a required identity, location, nationality, date, work-version, comparison, or multi-hop bridge is "
    "absent. An author is not a publisher, a performer is not a song's subject, a work's country is not its "
    "director's nationality, and a filming location is not automatically a narrative location. If the draft is I, "
    "change it to non-I only when you can cite direct answer evidence or one transparent operation: a win-loss "
    "field, date-range endpoint, comparison of explicit values, explicit acronym expansion, negative yes/no fact, "
    "or explicit chain ownership. If the question is malformed or omits a version and passages support multiple "
    "plausible answers, use A rather than I. Distinguish competing answers from different predicates or compatible "
    "precision. In reason begin with 'keep draft' or 'override draft', then give the exact passage-grounded cause."
)

DEBATE_ARBITER_SYSTEM_PROMPT = (
    "You are the final arbiter in an evidence-status debate. The user supplies the unchanged Original question and "
    "Search evidence. Three untrusted drafts in this system message represent a neutral judge, an insufficient-"
    "evidence prosecutor, and a non-insufficient defender. Draft text is not evidence and vote counts are irrelevant. "
    + OUTPUT_CONTRACT_STRICT
    + " Recheck the exact entity, predicate, scope, and each bridge in the passages. Use I only for zero complete "
    "answer candidates, S for exactly one, and A for multiple incompatible candidates fitting an underspecified "
    "question. Calibrate with these entirely fictional debate examples:\n"
    "- Fictional Q asks who distributed Amber Lake. Evidence only says Mira Sol directed it. Prosecutor says I; "
    "defender proposes Mira Sol. Final I, because directed is not distributed.\n"
    "- Fictional Q asks the Owls' wins. Evidence says 41-21. Prosecutor says the word wins is absent; defender says "
    "41. Final S with 41, because a record field is direct evidence.\n"
    "- Fictional Q asks the unqualified Nova Pad release date. Evidence gives Mini in 2012 and Pro in 2015. "
    "Prosecutor says I; defender picks 2012. Final A, because two versions fit.\n"
    "- Fictional Q asks which vessel is older, Orin or Pela. Evidence dates only Orin. Prosecutor says I; defender "
    "uses Orin's date. Final I, because one comparison operand is absent.\n"
    "Allow explicit cross-passage identity, negative yes/no facts, acronym expansion, ownership chains, range "
    "endpoints, record fields, and stated comparisons. Reject related-predicate, entity, version, nationality, and "
    "location substitutions. In reason identify which draft claim is passage-supported and the decisive bridge."
)

ENTAILMENT_CERTIFICATE_SYSTEM_PROMPT = (
    "You are a no-thinking evidence entailment verifier for factoid QA. Judge only the Original question; sub_query "
    "is retrieval history. "
    + OUTPUT_CONTRACT_STRICT
    + " Search the passages for a candidate answer, then build a minimal certificate with three fields in the "
    "reason: E=target entity resolution, P=exact requested predicate, B=all required bridges and scopes. Each field "
    "must cite passage numbers and be supported by their text; memorized knowledge and merely plausible facts are "
    "forbidden. If any required certificate field is absent, use insufficient_evidence. If one complete certificate "
    "exists, use supported_answer. If two or more incompatible complete certificates equally fit the question, use "
    "ambiguous_evidence. Directly reading a record field, range endpoint, explicit acronym, negative yes/no fact, "
    "ownership chain, or comparison of stated values is allowed. Do not demand exact wording when the passages "
    "explicitly entail the relation. Keep the certificate under 60 words."
)

MISSING_BRIDGE_AUDITOR_SYSTEM_PROMPT = (
    "You are a no-thinking missing-evidence auditor for factoid QA. Judge only the Original question against Search "
    "evidence; sub_query is retrieval history. "
    + OUTPUT_CONTRACT_STRICT
    + " Attempt to falsify answerability. Identify the strongest candidate visible in the passages, then check the "
    "target entity, exact predicate, explicit scope, and every multi-hop bridge. Use insufficient_evidence only when "
    "you can name a specific required fact that no passage supplies. Do not call ordinary textual entailment a gap: "
    "explicit cross-passage identity, record fields, range endpoints, acronym expansion, negative yes/no evidence, "
    "ownership chains, and stated comparisons are sufficient. Conversely, reject substitutions between related "
    "entities, predicates, work versions, places, dates, professions, or nationalities. If no concrete gap remains, "
    "use supported_answer for one candidate or ambiguous_evidence for multiple incompatible candidates. In reason "
    "write 'gap=' followed by the exact absent fact, or 'gap=none' followed by passage citations."
)

EVIDENCE_ONLY_BALANCED_SYSTEM_PROMPT = (
    BASELINE_SYSTEM_PROMPT
    + " Judge I versus non-I before distinguishing S from A. Use insufficient_evidence only when no complete answer "
    "to the Original question is supported. Treat descriptive clauses in the Original question as the target's "
    "given context; do not demand that passages restate every premise unless they identify a conflicting entity. "
    "Allow explicit cross-passage links and direct reading of ranges, records, acronyms, comparisons, ownership, and "
    "negative yes/no facts. Never replace the requested entity, predicate, version, place, date, profession, or "
    "nationality with a related one. One complete candidate is supported_answer; multiple incompatible candidates "
    "that fit an underspecified question are ambiguous_evidence. Keep reason to one short sentence and answer to the "
    "shortest span."
)

QUESTION_TAIL_FOCUS_SYSTEM_PROMPT = (
    BASELINE_SYSTEM_PROMPT
    + " The Original question is repeated after all passages. Treat that final repeated question as the sole target "
    "of the judgment. Passage titles and round markers are evidence organization only. Do not answer a related "
    "retrieval intent. Keep reason to one short sentence and answer to the shortest supported span."
)

QUESTION_TAIL_ANSWER_ALIGNMENT_SYSTEM_PROMPT = (
    BASELINE_SYSTEM_PROMPT
    + " For supported_answer, infer the answer type requested by the final repeated Original question, then copy "
    "the shortest passage span that exactly fills that type and predicate. Preserve the passage's canonical "
    "spelling, full proper name, date, number, or title; do not return a description, possessive reformulation, "
    "surrounding sentence, or a related entity. Do not remove words needed to distinguish the answer from another "
    "entity. This answer-extraction rule must not change whether the evidence is sufficient or ambiguous."
)
GOLD_SUPPORT_CHECK_SYSTEM_PROMPT = (
    "You are an evidence-grounded judge for factoid QA. The user supplies an Original question, a Reference gold "
    "answer, and Search evidence. The gold is only a candidate to verify and is never evidence. Do not claim the gold "
    "is supported unless the Search evidence establishes the exact Original-question relation. "
    + OUTPUT_CONTRACT_STRICT
    + " Identify the target entity, predicate, scope, and complete candidates. If evidence completely supports the "
    "gold and no equally matching competitor exists, use supported_answer. If it supports the gold and one or more "
    "equally matching incompatible candidates, use ambiguous_evidence. If it does not support the gold but supports "
    "one complete different answer, use supported_answer with that evidence answer. If it supports multiple complete "
    "different answers, use ambiguous_evidence. If neither the gold nor any other complete answer is supported, use "
    "insufficient_evidence. Different predicates and incomplete bridge chains are not candidates."
)

GOLD_DECOUPLED_STATUS_ANSWER_SYSTEM_PROMPT = (
    "You are an evidence-grounded judge for factoid QA. The user supplies an Original question, a Reference gold "
    "answer, and Search evidence. The gold is an answer-normalization hypothesis, never evidence. "
    + OUTPUT_CONTRACT_STRICT
    + " Separate status judgment from answer normalization. Stage 1: decide status as if the Reference gold were "
    "hidden. Count complete candidates supported by the passages for the Original question's exact entity, "
    "predicate, scope, and required bridges. Zero candidates means insufficient_evidence; one means "
    "supported_answer; multiple equally matching incompatible candidates mean ambiguous_evidence. Do not use "
    "insufficient_evidence merely because the supplied gold is unsupported when a different complete answer is "
    "supported. Stage 2 runs only after supported_answer: if the evidence-supported candidate matches a Reference "
    "gold answer or a clearly equivalent alias, return the shortest supported span in that gold-answer style. If "
    "the evidence supports a different answer, return that evidence answer instead of copying the gold. Never let "
    "the gold create a missing bridge, complete an unsupported relation, or change the Stage-1 status. In reason, "
    "briefly state the complete-candidate count and whether the supported final answer matches the gold hypothesis."
)

GOLD_COMPACT_BALANCED_SYSTEM_PROMPT = (
    "Judge the Original question only from Search evidence. The Reference gold answer is a hypothesis, not evidence. "
    + OUTPUT_CONTRACT_BASE
    + " Count complete evidence-supported answers for the exact requested relation: zero is insufficient_evidence, "
    "one is supported_answer, and multiple incompatible matches are ambiguous_evidence. A gold mention without the "
    "relation is incomplete. If one different answer is supported, return it as supported_answer rather than I. If "
    "the supported answer matches the gold or an alias, return its shortest supported gold-style span. Never invent "
    "a predicate, identity, scope, or bridge. Keep reason to one short sentence."
)

GOLD_I_GUARD_SYSTEM_PROMPT = (
    "You are a strict evidence-grounded judge. Judge the accumulated Search evidence against the Original question. "
    "The Reference gold answer is a hypothesis to check, not a fact and not permission to use outside knowledge. "
    + OUTPUT_CONTRACT_STRICT
    + " In reason explicitly report: target predicate; whether the gold is fully supported and by which passages; "
    "all other complete candidates; and the smallest missing bridge if none is complete. A candidate is complete only "
    "when passages cover the exact entity, predicate, scope, and every required bridge. Allow direct reading and one "
    "transparent passage-local inference such as a record number, date endpoint, explicit comparison, acronym, or "
    "single ownership link. Then count complete candidates: 0 -> insufficient_evidence; 1 -> supported_answer; 2+ "
    "equally matching incompatible candidates -> ambiguous_evidence. Do not turn different predicates, compatible "
    "precision, or members of one requested answer set into ambiguity."
)

GOLD_BINARY_SUPPORT_SYSTEM_PROMPT = (
    "You are a strict evidence verifier for factoid QA. The task is specifically to decide whether the Search "
    "evidence supports the supplied Reference gold answer as the answer to the Original question. The gold is a "
    "hypothesis, never evidence and never permission to use memorized facts. Sub_query is retrieval history. "
    + OUTPUT_CONTRACT_STRICT
    + " First identify the Original question's exact entity, predicate, scope, and every multi-hop bridge. Then test "
    "whether passages establish that complete relation to at least one Reference gold answer or a clearly equivalent "
    "alias. If no gold answer is fully supported, output insufficient_evidence even when the passages support a "
    "different answer. Do not replace the requested predicate, entity, work version, location, date, comparison, or "
    "bridge with a related fact. A lexical mention of the gold is not enough unless it participates in the requested "
    "relation. If a gold answer is fully supported and no equally matching incompatible candidate exists, output "
    "supported_answer. If a gold answer is supported but the underspecified question and evidence also support one "
    "or more incompatible answers, output ambiguous_evidence. S versus A is secondary; never use I when the gold "
    "relation is established. Allow only transparent evidence operations such as explicit cross-passage identity "
    "links, acronym expansion, a birth-death endpoint, a win-loss record, or comparison of stated dates. In reason "
    "state 'gold supported: yes' or 'gold supported: no', cite the decisive passages, and name the smallest missing "
    "relation when no. For supported_answer, return the shortest supported gold alias."
)


@dataclass(frozen=True)
class PromptVariant:
    name: str
    system_prompt: str
    include_gold: bool
    family: str
    description: str
    layout: str = "v2"


PROMPT_VARIANTS: dict[str, PromptVariant] = {
    "baseline_historical_v1": PromptVariant(
        name="baseline_historical_v1",
        system_prompt=BASELINE_SYSTEM_PROMPT,
        include_gold=False,
        family="layout_ablation",
        description="Historical 500-run system instruction and flat v1 user evidence layout.",
        layout="v1",
    ),
    "baseline_evidence_only_v2": PromptVariant(
        name="baseline_evidence_only_v2",
        system_prompt=BASELINE_SYSTEM_PROMPT,
        include_gold=False,
        family="layout_ablation",
        description="Current hierarchical passage layout with retrieval sub_query strings removed.",
        layout="evidence_only",
    ),
    "candidate_count_evidence_only": PromptVariant(
        name="candidate_count_evidence_only",
        system_prompt=CANDIDATE_COUNT_SYSTEM_PROMPT,
        include_gold=False,
        family="layout_ablation",
        description="A1 candidate-count instruction with sub_query strings removed.",
        layout="evidence_only",
    ),
    "baseline_text_only_v2": PromptVariant(
        name="baseline_text_only_v2",
        system_prompt=BASELINE_SYSTEM_PROMPT,
        include_gold=False,
        family="layout_ablation",
        description="Current layout without sub_query strings or passage titles.",
        layout="text_only",
    ),
    "baseline_top3_evidence_only_v2": PromptVariant(
        name="baseline_top3_evidence_only_v2",
        system_prompt=BASELINE_SYSTEM_PROMPT,
        include_gold=False,
        family="layout_ablation",
        description="Evidence-only layout limited to the first three passages per round.",
        layout="evidence_top3",
    ),
    "baseline_flat_evidence_only_v2": PromptVariant(
        name="baseline_flat_evidence_only_v2",
        system_prompt=BASELINE_SYSTEM_PROMPT,
        include_gold=False,
        family="layout_ablation",
        description="Evidence-only passages flattened without round headings.",
        layout="evidence_flat",
    ),
    "baseline_question_tail_evidence_only_v2": PromptVariant(
        name="baseline_question_tail_evidence_only_v2",
        system_prompt=BASELINE_SYSTEM_PROMPT,
        include_gold=False,
        family="layout_ablation",
        description="Evidence-only layout that repeats the Original question immediately before generation.",
        layout="evidence_question_tail",
    ),
    "question_tail_answer_alignment_v3": PromptVariant(
        name="question_tail_answer_alignment_v3",
        system_prompt=QUESTION_TAIL_ANSWER_ALIGNMENT_SYSTEM_PROMPT,
        include_gold=False,
        family="answer_alignment_ablation",
        description="Successful question-tail evidence-only layout with exact answer-type span extraction.",
        layout="evidence_question_tail",
    ),
    "baseline_question_tail_v2": PromptVariant(
        name="baseline_question_tail_v2",
        system_prompt=BASELINE_SYSTEM_PROMPT,
        include_gold=False,
        family="layout_ablation",
        description="Current v2 layout retaining sub_query and repeating Original question at the end.",
        layout="v2_question_tail",
    ),
    "baseline_delimited_evidence_only_v2": PromptVariant(
        name="baseline_delimited_evidence_only_v2",
        system_prompt=BASELINE_SYSTEM_PROMPT,
        include_gold=False,
        family="layout_ablation",
        description="Evidence-only layout with explicit passage start/end delimiters.",
        layout="evidence_delimited",
    ),
    "balanced_evidence_only_v2": PromptVariant(
        name="balanced_evidence_only_v2",
        system_prompt=EVIDENCE_ONLY_BALANCED_SYSTEM_PROMPT,
        include_gold=False,
        family="instruction_layout_ablation",
        description="Concise I/non-I policy combined with the evidence-only layout.",
        layout="evidence_only",
    ),
    "baseline_top3_question_tail_v2": PromptVariant(
        name="baseline_top3_question_tail_v2",
        system_prompt=BASELINE_SYSTEM_PROMPT,
        include_gold=False,
        family="layout_ablation",
        description="Top-three evidence-only passages with Original question repeated at the end.",
        layout="evidence_top3_question_tail",
    ),
    "baseline_text_question_tail_v2": PromptVariant(
        name="baseline_text_question_tail_v2",
        system_prompt=BASELINE_SYSTEM_PROMPT,
        include_gold=False,
        family="layout_ablation",
        description="Title-free evidence-only passages with Original question repeated at the end.",
        layout="evidence_text_question_tail",
    ),
    "baseline_delimited_question_tail_v2": PromptVariant(
        name="baseline_delimited_question_tail_v2",
        system_prompt=BASELINE_SYSTEM_PROMPT,
        include_gold=False,
        family="layout_ablation",
        description="Delimited evidence-only passages with Original question repeated at the end.",
        layout="evidence_delimited_question_tail",
    ),
    "focused_question_tail_evidence_only_v2": PromptVariant(
        name="focused_question_tail_evidence_only_v2",
        system_prompt=QUESTION_TAIL_FOCUS_SYSTEM_PROMPT,
        include_gold=False,
        family="instruction_layout_ablation",
        description="Evidence-only question-tail layout with an explicit sole-target system instruction.",
        layout="evidence_question_tail",
    ),
    "baseline_question_tail_only_v2": PromptVariant(
        name="baseline_question_tail_only_v2",
        system_prompt=BASELINE_SYSTEM_PROMPT,
        include_gold=False,
        family="layout_ablation",
        description="Evidence first, with Original question appearing only once at the end.",
        layout="evidence_question_tail_only",
    ),
    "baseline_tagged_question_tail_v2": PromptVariant(
        name="baseline_tagged_question_tail_v2",
        system_prompt=BASELINE_SYSTEM_PROMPT,
        include_gold=False,
        family="layout_ablation",
        description="Evidence-only question-tail layout using explicit original_question tags.",
        layout="evidence_tagged_question_tail",
    ),
    "baseline_decision_question_tail_v2": PromptVariant(
        name="baseline_decision_question_tail_v2",
        system_prompt=BASELINE_SYSTEM_PROMPT,
        include_gold=False,
        family="layout_ablation",
        description="Evidence-only question-tail layout with a compact candidate-count reminder.",
        layout="evidence_decision_question_tail",
    ),
    "baseline_current_v2": PromptVariant(
        name="baseline_current_v2",
        system_prompt=BASELINE_SYSTEM_PROMPT,
        include_gold=False,
        family="instruction_only",
        description="Current production system instruction with the current v2 user layout.",
    ),
    "candidate_count": PromptVariant(
        name="candidate_count",
        system_prompt=CANDIDATE_COUNT_SYSTEM_PROMPT,
        include_gold=False,
        family="instruction_only",
        description="Classify by exact-target complete-candidate count.",
    ),
    "candidate_count_i_guard": PromptVariant(
        name="candidate_count_i_guard",
        system_prompt=CANDIDATE_COUNT_I_GUARD_SYSTEM_PROMPT,
        include_gold=False,
        family="instruction_only",
        description="Candidate count plus missing-bridge and bounded-inference guards for I.",
    ),
    "candidate_count_i_guard_compact": PromptVariant(
        name="candidate_count_i_guard_compact",
        system_prompt=CANDIDATE_COUNT_I_GUARD_COMPACT_SYSTEM_PROMPT,
        include_gold=False,
        family="instruction_only",
        description="Compact form of the I-guard decision procedure.",
    ),
    "binary_i_gate": PromptVariant(
        name="binary_i_gate",
        system_prompt=BINARY_I_GATE_SYSTEM_PROMPT,
        include_gold=False,
        family="instruction_only",
        description="Two-stage I/non-I gate followed by the tolerated S/A distinction.",
    ),
    "candidate_count_balanced": PromptVariant(
        name="candidate_count_balanced",
        system_prompt=CANDIDATE_COUNT_BALANCED_SYSTEM_PROMPT,
        include_gold=False,
        family="instruction_only",
        description="Candidate count with balanced exact-predicate, entailment, negative-answer, and ambiguity guards.",
    ),
    "calibrated_fewshot": PromptVariant(
        name="calibrated_fewshot",
        system_prompt=CALIBRATED_FEWSHOT_SYSTEM_PROMPT,
        include_gold=False,
        family="instruction_only_fewshot",
        description="Synthetic boundary examples calibrate I/non-I without benchmark-label leakage.",
    ),
    "meta_arbiter": PromptVariant(
        name="meta_arbiter",
        system_prompt=META_ARBITER_SYSTEM_PROMPT,
        include_gold=False,
        family="multi_draft_meta",
        description="Independently reconcile untrusted no-gold draft judgments from the same evidence.",
    ),
    "draft_critic": PromptVariant(
        name="draft_critic",
        system_prompt=DRAFT_CRITIC_SYSTEM_PROMPT,
        include_gold=False,
        family="multi_draft_meta",
        description="Conservatively verify and only concretely override one A1 first-pass draft.",
    ),
    "debate_arbiter": PromptVariant(
        name="debate_arbiter",
        system_prompt=DEBATE_ARBITER_SYSTEM_PROMPT,
        include_gold=False,
        family="multi_draft_meta",
        description="Resolve neutral/prosecutor/defender drafts using only fictional calibration debates.",
    ),
    "entailment_certificate": PromptVariant(
        name="entailment_certificate",
        system_prompt=ENTAILMENT_CERTIFICATE_SYSTEM_PROMPT,
        include_gold=False,
        family="structured_binary_worker",
        description="Require a passage-grounded entity/predicate/bridge certificate.",
    ),
    "missing_bridge_auditor": PromptVariant(
        name="missing_bridge_auditor",
        system_prompt=MISSING_BRIDGE_AUDITOR_SYSTEM_PROMPT,
        include_gold=False,
        family="structured_binary_worker",
        description="Falsify answerability and emit I only for a concrete absent fact.",
    ),
    "gold_support_check": PromptVariant(
        name="gold_support_check",
        system_prompt=GOLD_SUPPORT_CHECK_SYSTEM_PROMPT,
        include_gold=True,
        family="gold_aware",
        description="Treat gold as a candidate whose evidence support must be verified.",
    ),
    "gold_support_question_tail_v3": PromptVariant(
        name="gold_support_question_tail_v3",
        system_prompt=GOLD_SUPPORT_CHECK_SYSTEM_PROMPT,
        include_gold=True,
        family="gold_aware_layout_ablation",
        description="Gold-hypothesis verification on the successful no-subquery question-tail layout.",
        layout="gold_evidence_question_tail",
    ),
    "gold_support_evidence_only_v3": PromptVariant(
        name="gold_support_evidence_only_v3",
        system_prompt=GOLD_SUPPORT_CHECK_SYSTEM_PROMPT,
        include_gold=True,
        family="gold_aware_layout_ablation",
        description="Gold-hypothesis verification without sub-query strings or a question tail.",
        layout="gold_evidence_only",
    ),
    "gold_decoupled_status_answer_v3": PromptVariant(
        name="gold_decoupled_status_answer_v3",
        system_prompt=GOLD_DECOUPLED_STATUS_ANSWER_SYSTEM_PROMPT,
        include_gold=True,
        family="gold_aware_instruction_ablation",
        description="Judge status without gold, then use supported gold only to normalize an S answer.",
        layout="gold_evidence_only",
    ),
    "gold_compact_balanced_v3": PromptVariant(
        name="gold_compact_balanced_v3",
        system_prompt=GOLD_COMPACT_BALANCED_SYSTEM_PROMPT,
        include_gold=True,
        family="gold_aware_instruction_ablation",
        description="Compact balanced gold verifier on the winning gold-aware layout.",
        layout="gold_evidence_only",
    ),
    "gold_support_subquery_question_tail_v3": PromptVariant(
        name="gold_support_subquery_question_tail_v3",
        system_prompt=GOLD_SUPPORT_CHECK_SYSTEM_PROMPT,
        include_gold=True,
        family="gold_aware_layout_ablation",
        description="Gold-hypothesis verification retaining sub-query strings and adding a question tail.",
        layout="gold_v2_question_tail",
    ),
    "gold_i_guard": PromptVariant(
        name="gold_i_guard",
        system_prompt=GOLD_I_GUARD_SYSTEM_PROMPT,
        include_gold=True,
        family="gold_aware",
        description="Gold support check with explicit complete-candidate and missing-bridge audit.",
    ),
    "gold_i_guard_evidence_only_v3": PromptVariant(
        name="gold_i_guard_evidence_only_v3",
        system_prompt=GOLD_I_GUARD_SYSTEM_PROMPT,
        include_gold=True,
        family="gold_aware_instruction_ablation",
        description="Historical strict gold I-guard on the new winning gold-aware layout.",
        layout="gold_evidence_only",
    ),
    "gold_binary_support": PromptVariant(
        name="gold_binary_support",
        system_prompt=GOLD_BINARY_SUPPORT_SYSTEM_PROMPT,
        include_gold=True,
        family="gold_aware",
        description="Use I exactly when the evidence does not support the reference gold relation.",
    ),
    "gold_binary_support_evidence_only_v3": PromptVariant(
        name="gold_binary_support_evidence_only_v3",
        system_prompt=GOLD_BINARY_SUPPORT_SYSTEM_PROMPT,
        include_gold=True,
        family="gold_aware_instruction_ablation",
        description="Historical gold binary-support gate on the new winning gold-aware layout.",
        layout="gold_evidence_only",
    ),
}


def _indent(value: Any, spaces: int) -> str:
    prefix = " " * spaces
    return prefix + str(value or "").replace("\n", f"\n{prefix}")


def build_user_prompt(case: dict[str, Any], *, include_gold: bool) -> str:
    """Render the current production v2 user layout, optionally adding gold."""

    lines = ["   Original question:", _indent(case["question"], 6)]
    if include_gold:
        lines.extend(
            [
                "",
                "   Reference gold answer:",
                _indent(json.dumps(case.get("gold_answers") or [], ensure_ascii=False), 6),
            ]
        )
    lines.extend(["", "   Search evidence:"])
    evidence_steps = case.get("evidence_steps") or []
    if not evidence_steps:
        lines.extend(["", "      (no search evidence provided)"])
    for idx, step in enumerate(evidence_steps, start=1):
        lines.extend(
            [
                "",
                f"      Round {idx}:",
                f"         sub_query: {step.get('sub_query') or ''}",
                "         retrieved contents:",
            ]
        )
        for doc_idx, doc in enumerate((step.get("docs") or [])[:5], start=1):
            title = doc.get("title") or doc.get("doc_id") or f"doc-{doc_idx}"
            text = doc.get("contents") or doc.get("text") or doc.get("passage") or ""
            lines.append(f"            [{doc_idx}] {title}")
            lines.append(_indent(text, 15))
    lines.extend(
        [
            "",
            "   Now output the final result directly. "
            "Do not analyze the instruction. Do not repeat rules. Begin with <reason>.",
        ]
    )
    return "\n".join(lines)


def build_user_prompt_v1(case: dict[str, Any]) -> str:
    """Render the historical flat user layout used by the 500-sample run."""

    lines = [f"Original question:\n{case['question']}", "", "Search evidence:"]
    evidence_steps = case.get("evidence_steps") or []
    if not evidence_steps:
        lines.append("(no search evidence provided)")
    for index, step in enumerate(evidence_steps, start=1):
        lines.append(f"\nRound {index} sub_query:\n{step.get('sub_query') or ''}")
        for doc_index, doc in enumerate((step.get("docs") or [])[:5], start=1):
            title = doc.get("title") or doc.get("doc_id") or f"doc-{doc_index}"
            contents = doc.get("contents") or doc.get("text") or doc.get("passage") or ""
            lines.append(f"[{doc_index}] {title}\n{contents}")
    lines.append(
        "\nNow output the final result directly. "
        "Do not analyze the instruction. Do not repeat rules. Begin with <reason>."
    )
    return "\n".join(lines)


def build_user_prompt_evidence_only(
    case: dict[str, Any],
    *,
    include_titles: bool = True,
    max_docs: int = 5,
    include_rounds: bool = True,
    repeat_question: bool = False,
    delimited: bool = False,
    omit_initial_question: bool = False,
    tagged_question: bool = False,
    decision_reminder: bool = False,
    include_gold: bool = False,
) -> str:
    """Render all retrieved passages while omitting potentially distracting sub-query strings."""

    lines = []
    if not omit_initial_question:
        if tagged_question:
            lines.extend(["   <original_question>", _indent(case["question"], 6), "   </original_question>"])
        else:
            lines.extend(["   Original question:", _indent(case["question"], 6)])
        if include_gold:
            lines.extend(
                [
                    "",
                    "   Reference gold answer:",
                    _indent(json.dumps(case.get("gold_answers") or [], ensure_ascii=False), 6),
                ]
            )
        lines.append("")
    lines.append("   Search evidence:")
    evidence_steps = case.get("evidence_steps") or []
    if not evidence_steps:
        lines.extend(["", "      (no search evidence provided)"])
    flat_doc_index = 0
    for index, step in enumerate(evidence_steps, start=1):
        if include_rounds:
            lines.extend(["", f"      Round {index}:", "         retrieved contents:"])
        for doc_index, doc in enumerate((step.get("docs") or [])[:max_docs], start=1):
            flat_doc_index += 1
            title = doc.get("title") or doc.get("doc_id") or f"doc-{doc_index}"
            contents = doc.get("contents") or doc.get("text") or doc.get("passage") or ""
            display_index = doc_index if include_rounds else flat_doc_index
            if delimited:
                lines.append(f"            <passage id=\"{display_index}\">")
                if include_titles:
                    lines.append(f"               title: {title}")
            elif include_titles:
                lines.append(f"            [{display_index}] {title}")
            else:
                lines.append(f"            [{display_index}]")
            lines.append(_indent(contents, 15))
            if delimited:
                lines.append("            </passage>")
    if repeat_question:
        if tagged_question:
            lines.extend(
                ["", "   <original_question>", _indent(case["question"], 6), "   </original_question>"]
            )
        else:
            lines.extend(["", "   Original question to judge:", _indent(case["question"], 6)])
    if decision_reminder:
        lines.extend(
            [
                "",
                "   Decision reminder: zero complete answer candidates means insufficient_evidence; "
                "one means supported_answer; multiple incompatible candidates means ambiguous_evidence.",
            ]
        )
    lines.extend(
        [
            "",
            "   Now output the final result directly. "
            "Do not analyze the instruction. Do not repeat rules. Begin with <reason>.",
        ]
    )
    return "\n".join(lines)


def build_user_prompt_v2_question_tail(
    case: dict[str, Any], *, include_gold: bool = False
) -> str:
    prompt = build_user_prompt(case, include_gold=include_gold)
    prompt = prompt.rsplit("\n   Now output the final result directly.", 1)[0]
    return (
        prompt
        + "\n\n   Original question to judge:\n"
        + _indent(case["question"], 6)
        + "\n\n   Now output the final result directly. "
        "Do not analyze the instruction. Do not repeat rules. Begin with <reason>."
    )


def build_messages(case: dict[str, Any], variant_name: str) -> list[dict[str, str]]:
    try:
        variant = PROMPT_VARIANTS[variant_name]
    except KeyError as exc:
        available = ", ".join(sorted(PROMPT_VARIANTS))
        raise ValueError(f"Unknown prompt variant {variant_name!r}; available: {available}") from exc
    layout_builders = {
        "v1": lambda: build_user_prompt_v1(case),
        "v2": lambda: build_user_prompt(case, include_gold=variant.include_gold),
        "v2_question_tail": lambda: build_user_prompt_v2_question_tail(case),
        "gold_v2_question_tail": lambda: build_user_prompt_v2_question_tail(
            case, include_gold=True
        ),
        "evidence_only": lambda: build_user_prompt_evidence_only(case),
        "text_only": lambda: build_user_prompt_evidence_only(case, include_titles=False),
        "evidence_top3": lambda: build_user_prompt_evidence_only(case, max_docs=3),
        "evidence_flat": lambda: build_user_prompt_evidence_only(case, include_rounds=False),
        "evidence_question_tail": lambda: build_user_prompt_evidence_only(
            case, repeat_question=True
        ),
        "gold_evidence_question_tail": lambda: build_user_prompt_evidence_only(
            case, repeat_question=True, include_gold=True
        ),
        "gold_evidence_only": lambda: build_user_prompt_evidence_only(
            case, include_gold=True
        ),
        "evidence_delimited": lambda: build_user_prompt_evidence_only(case, delimited=True),
        "evidence_top3_question_tail": lambda: build_user_prompt_evidence_only(
            case, max_docs=3, repeat_question=True
        ),
        "evidence_text_question_tail": lambda: build_user_prompt_evidence_only(
            case, include_titles=False, repeat_question=True
        ),
        "evidence_delimited_question_tail": lambda: build_user_prompt_evidence_only(
            case, delimited=True, repeat_question=True
        ),
        "evidence_question_tail_only": lambda: build_user_prompt_evidence_only(
            case, omit_initial_question=True, repeat_question=True
        ),
        "evidence_tagged_question_tail": lambda: build_user_prompt_evidence_only(
            case, repeat_question=True, tagged_question=True
        ),
        "evidence_decision_question_tail": lambda: build_user_prompt_evidence_only(
            case, repeat_question=True, decision_reminder=True
        ),
    }
    try:
        user_content = layout_builders[variant.layout]()
    except KeyError as exc:
        raise ValueError(f"Unknown user prompt layout {variant.layout!r}") from exc
    return [
        {"role": "system", "content": variant.system_prompt},
        {"role": "user", "content": user_content},
    ]
