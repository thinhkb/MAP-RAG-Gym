from map_rag_gym.core.pipeline import MAPRAGGym
from map_rag_gym.router.rule_based import RuleBasedRouter
from map_rag_gym.utils.io import write_json

pipe = MAPRAGGym("data/sample/corpus.json", llm_provider="dummy")
router = RuleBasedRouter()
examples = [
    ("Who wrote Pride and Prejudice?", "Jane Austen"),
    ("Which novel by the author of Pride and Prejudice was published posthumously?", "Persuasion"),
    ("Compare California and Japan in terms of GDP.", "California has a GDP larger than Japan in this sample corpus"),
]
runs = []
for q, a in examples:
    decision = router.decide(q)
    res = pipe.run(q, a, decision.workflow_id, planner_reason=decision.reason)
    runs.append(res.to_dict())
    print(q, "->", decision.workflow_id, res.final_scores)
write_json("outputs/phase2_router.json", runs)
print("Saved outputs/phase2_router.json")
