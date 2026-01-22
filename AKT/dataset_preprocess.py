import pandas as pd

from sklearn.model_selection import GroupShuffleSplit

path = ""

data = pd.read_csv(path, encoding="latin1")

df = data[["user_id", "problem_id", "skill_id", "correct", "order_id"]]
df = df.dropna(subset=["skill_id"]) # remove rows with no concept

df["skill_id"] = df["skill_id"].astype(int)
df["correct"] = df["correct"].astype(int)

df = df.sort_values(["user_id", "order_id"])

q_map = {q:i for i,q in enumerate(df["problem_id"].unique())}
c_map = {c:i for i,c in enumerate(df["skill_id"].unique())}

df["q"] = df["problem_id"].map(q_map)
df["c"] = df["skill_id"].map(c_map)
df["r"] = df["correct"]


gss = GroupShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=42
)

groups = df["user_id"].values
X = df.index.values
y = df["correct"].values

train_idx, test_idx = next(gss.split(X, y, groups=groups))

df_train = df.iloc[train_idx].copy()
df_test  = df.iloc[test_idx].copy()


def build_student_seqs(d):
    seqs = []
    for uid, u_df in d.groupby("user_id"):
        q = u_df["q"].values
        c = u_df["c"].values
        r = u_df["r"].values
        if len(q) >= 2:   # need history + target
            seqs.append((q, c, r))
    return seqs


train_student_seqs = build_student_seqs(df_train)
test_student_seqs  = build_student_seqs(df_test)

context_size = 20 # including query


def build_samples(student_seqs):
    samples = []
    for q, c, r in student_seqs:
        L = len(q)
        for i in range(0, L - context_size + 1):
            q_hist = q[i:i+context_size-1]
            c_hist = c[i:i+context_size-1]
            r_hist = r[i:i+context_size-1]

            q_query = q[i+context_size-1]
            c_query = c[i+context_size-1]
            r_target = r[i+context_size-1]

            samples.append((q_hist, c_hist, r_hist, q_query, c_query, r_target))
    return samples


train_samples = build_samples(train_student_seqs)
test_samples  = build_samples(test_student_seqs)
