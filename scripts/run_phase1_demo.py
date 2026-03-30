from map_rag_gym.core.pipeline import MAPRAGGym
from map_rag_gym.core.workflows import WORKFLOWS
from map_rag_gym.utils.io import write_json

pipe = MAPRAGGym("data/sample/corpus.json", llm_provider="dummy")
question = "Which novel by the author of Pride and Prejudice was published posthumously?"
gold = "Persuasion"
runs = []
for wf in WORKFLOWS:
    res = pipe.run(question, gold, wf)
    runs.append(res.to_dict())
    print(wf, res.final_answer, res.final_scores)
write_json("outputs/phase1_demo.json", runs)
print("Saved outputs/phase1_demo.json")
