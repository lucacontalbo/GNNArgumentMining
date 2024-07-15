from kogito.models.bart.comet import COMETBART
from kogito.inference import CommonsenseInference
import pandas as pd

class GraphCreator:

    def __init__(self):
        self.model = COMETBART.from_pretrained("mismayil/comet-bart-ai2")
        self.csi = CommonsenseInference(language="en_core_web_sm")
    
    def run(self, sentence, num_hops=2):
        # Run inference
        kgraph = self.csi.infer(sentence, self.model)
        graph_as_list = []
        for knowl in kgraph:
            for tail in knowl.tails:
                graph_as_list.append((knowl.head, knowl.relation, tail))
                graph_as_list.extend(self.run(tail))
        
        return graph_as_list
    
    def run_dataset(self, path, sent1_name, sent2_name, index_col=None):
        df = pd.read_csv(path, index_col=index_col)

        for i,row in df.iterrows():
            df.at[i, "sent1_graph"] = list(set(self.run(row[sent1_name])))
            df.at[i, "sent2_graph"] = list(set(self.run(row[sent2_name])))
        
        splitted_path = path.split(".")
        df.to_csv(".".join(splitted_path[:-1])+"_new."+splitted_path[-1])

if __name__ == "__main__":
    gc = GraphCreator()
    se = "./data/student_essay.csv"
    db = "./data/debate.csv"
    marg = "./data/presidential_final.csv"

    gc.run_dataset(se, "Arg1", "Arg2", index_col=0)
    gc.run_dataset(db, "Arg1", "Arg2", index_col=0)
    gc.run_dataset(marg, "Arg1", "Arg2", index_col=None)