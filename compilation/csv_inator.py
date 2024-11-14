class CSVInator:
    def __init__(self,data1,data2,function) -> None:
        self.data1 = data1
        self.data2 = data2
        self.function = function
        
    def __call__(self,filepath,name1="impath1",name2="impath2",fname="similarity"):
        with open(filepath,"w") as file:
            file.write(f"{name1},{name2},{fname}\n")
            for d1,d2 in zip(self.data1,self.data2):
                file.write(f"{d1},{d2},{self.function(d1,d2)}\n")        