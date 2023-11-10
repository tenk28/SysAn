import numpy as np
import pandas as pd

data = pd.read_excel("/home/olehborysevych/Dev/Education/University/3 course/SA/lab2/SysAn/PrevCourse/Альтернативна_вибірка.xlsx")
data = data.drop("Unnamed: 0", axis=1)

for col in data.columns:
    if col == "x31":
        data[col] += np.random.randint(0, 2, len(data[col]))
    elif col[0] == "y":
        data[col] += np.random.normal(0, 50, len(data[col]))
    else:
        data[col] += np.random.normal(0, 0.1, len(data[col]))

data["y5"] = data["y4"] + np.random.normal(0, 0.1, len(data["y4"]))

data = data.reset_index()
data["index"] += 1
print(data)
data.to_excel("Альтернативна_вибірка.xlsx", index=False)
