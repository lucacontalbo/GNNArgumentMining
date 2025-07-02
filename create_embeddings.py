from kogito.models.bart.comet import COMETBART
from kogito.inference import CommonsenseInference
import pandas as pd
from tqdm import tqdm

class GraphCreator:

    def __init__(self):
        self.model = COMETBART.from_pretrained("mismayil/comet-bart-ai2")
        self.csi = CommonsenseInference(language="en_core_web_sm")
    
    def run(self, sentence, num_hops=1):
        if num_hops == 0:
            return -1
        kgraph = self.csi.infer(sentence, self.model)
        #print(sentence)
        #print(kgraph)
        graph_as_list = []
        for knowl in kgraph:
            if str(knowl.relation).lower().strip() not in ["causes", "hinderedby", "isafter", "isbefore"]:
                continue
            for tail in knowl.tails:
                graph_as_list.append((str(knowl.head), str(knowl.relation), str(tail)))
                el = self.run(tail, num_hops-1)
                if el == -1:
                    break
                graph_as_list.extend(self.run(tail, num_hops-1))
        
        return graph_as_list
    
    def run_dataset(self, path, sent1_name, sent2_name, index_col=None):
        df = pd.read_csv(path, index_col=index_col)
        df["sent1_graph"] = ["" for _ in range(len(df))]
        df["sent2_graph"] = ["" for _ in range(len(df))]

        for i,row in tqdm(df.iterrows()):
            df.at[i, "sent1_graph"] = list(set(self.run(row[sent1_name])))
            df.at[i, "sent2_graph"] = list(set(self.run(row[sent2_name])))
        
        splitted_path = path.split(".")
        df.to_csv(".".join(splitted_path[:-1])+"_new."+splitted_path[-1])

if __name__ == "__main__":
    gc = GraphCreator()
    se = "./data/student_essay.csv"
    db = "./data/debate.csv"
    marg = "./data/presidential_final.csv"

    #gc.run_dataset(se, "Arg1", "Arg2", index_col=0)
    gc.run_dataset(db, "Arg1", "Arg2", index_col=0)
    gc.run_dataset(marg, "Arg1", "Arg2", index_col=None)
