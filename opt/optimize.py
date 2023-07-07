import pandas as pd
import numpy as np
from functools import reduce
from datetime import date, datetime
from mip import Model, minimize, xsum

start = datetime.now()


def read_data():
    def conv_str_date(string):
        return date(
            int(string.split("/")[0]),
            int(string.split("/")[1]),
            int(string.split("/")[2]),
        )

    information_pd = pd.read_csv("csv/information.csv")
    schedule_pd = pd.read_csv("csv/chouseisan.csv")

    schedule_dict = {"◯": 2, "△": 1, "×": 0}

    information = {}
    for v in np.array(information_pd):
        v[3] = conv_str_date(v[3])
        information[v[0]] = list(v[1:])

    shape = np.array(schedule_pd)[1:, 1:-1].shape
    schedule = np.array(
        [schedule_dict[e] for e in np.array(schedule_pd)[1:, 1:-1].flatten()]
    ).reshape(shape)

    members = np.array(schedule_pd)[1:, 0]
    candidate = np.array([conv_str_date(e) for e in schedule_pd.columns[1:-1]])
    candidate_info = np.array(schedule_pd)[0, 1:-1]
    candidate_group = np.array([*map(lambda x: int(x.split("/")[0]), candidate_info)])
    candidate_todo = [
        *map(lambda x: x.split("/")[1:] if "/" in x else [], candidate_info)
    ]

    return information, schedule, members, candidate, candidate_group, candidate_todo


class VisitModel(Model):
    def add_vars(
        self, information, schedule, members, candidate, candidate_group, candidate_todo
    ):
        self.information = information
        self.schedule = schedule
        self.members = members
        self.candidate = candidate
        self.candidate_group = candidate_group
        self.candidate_todo = candidate_todo
        self.todo = list(set(reduce(lambda x, y: x + y, candidate_todo, [])))
        self.project_members = {}

        for m in self.members:
            try:
                pjs = information[m][3].split("/")
            except:
                continue
            for pj in pjs:
                if pj not in self.project_members:
                    self.project_members[pj] = [m]
                self.project_members[pj].append(m)

        self.X = np.array(
            [
                self.add_var(name=f"X_{i}", var_type="B")
                if schedule.flatten()[i]
                else 0
                for i in range(schedule.size)
            ]
        ).reshape(schedule.shape)

        self.y = [1 for _ in range(len(self.candidate))]

        for i in set(candidate_group):
            if np.count_nonzero(candidate_group == i) != 1:
                for j in np.where(candidate_group == i)[0]:
                    self.y[j] = self.add_var(name=f"y_{j}", var_type="B")

                self += xsum(self.y[j] for j in np.where(candidate_group == i)[0]) == 1

        self.a = self.add_var_tensor((len(members),), name="alpha", var_type="B")
        self.b = self.add_var_tensor((len(candidate),), name="beta", var_type="B")

    def _cons_driver(self):
        for d in range(len(self.candidate)):
            self += (
                xsum(
                    self.X[i, d] * self.information[m][1]
                    for i, m in enumerate(self.members)
                )
                >= 3 * self.y[d] - 2 * self.b[d]
            )

    def _cons_visiter(self):
        for d in range(len(self.candidate)):
            self += (
                xsum(self.X[i, d] for i in range(len(self.members)))
                == 7 * self.y[d] - self.b[d]
            )

    def _cons_visit_time(self):
        for m in range(len(self.members)):
            self += xsum(self.X[m, d] for d in range(len(self.candidate))) >= self.a[m]

    def _cons_senior(self):
        senior_ls = []
        for i, m in enumerate(self.members):
            if self.information[m][0] == 8:
                senior_ls.append(i)

        for d in range(len(self.candidate)):
            self += xsum(self.X[s, d] for s in senior_ls) >= self.y[d]

    def _cons_project(self):
        for i, todo_ls in enumerate(self.candidate_todo):
            for todo in todo_ls:
                self += (
                    xsum(
                        self.X[np.where(self.members == m)[0][0], i]
                        for m in self.project_members[todo]
                    )
                    >= self.y[i]
                )

    def set_objective(self):
        self.objective = minimize(
            1000 * xsum(y for y in self.y)
            - 10
            * xsum(
                self.X[m, d] * self.schedule[m, d]
                for m in range(len(self.members))
                for d in range(len(self.candidate))
            )
            - xsum(
                self.X[m, d]
                * (self.candidate[d] - self.information[self.members[m]][2]).days
                for m in range(len(self.members))
                for d in range(len(self.candidate))
            )
            + 10
            * xsum(
                y * (self.candidate[i] - date(2023, 1, 1)).days
                for i, y in enumerate(self.y)
            )
            - 100 * xsum(self.a[m] for m in range(len(self.members)))
            + 1000 * xsum(self.b[d] for d in range(len(self.candidate)))
        )

    def show_result(self):
        def opt(obj):
            try:
                return int(obj.x)
            except:
                return obj

        if self.objective_value == None:
            print("不能 (解なし)")
            exit(1)
        print(f"\n\n目的関数値\t: {int(self.objective_value)}")

        X_opt = np.array([opt(x) for x in self.X.flatten()]).reshape(
            self.schedule.shape
        )
        y_opt = np.array([opt(y) for y in self.y])
        a_opt = np.array([opt(a) for a in self.a])
        b_opt = np.array([opt(b) for b in self.b])

        self.simpl_date = np.array([f"{d.month}/{d.day}" for d in self.candidate])

        print("==" * 64)
        print(
            f"訪問回数\t: {sum(y_opt)} 回",
        )
        print(f"回答者\t\t: {len(self.members)} 人")
        print(f"訪問可能人数\t: {sum(a_opt)} 人")
        print(f"訪問日\t\t:", *self.simpl_date[y_opt == 1])
        print(f"先輩ドライバー\t:", *self.simpl_date[b_opt == 1])
        print("--" * 64)
        print("回答者別\n\n● : ドライバー\t◯ : 訪問あり\t× : 訪問なし\n")
        for m in range(len(self.members)):
            print(
                (members[m] + "      ")[:6] + "\t:",
                *np.where(
                    X_opt[m][y_opt == 1] == 1,
                    "●" if self.information[self.members[m]][1] > 0 else "◯",
                    "×",
                ),
                "\t",
                *self.simpl_date[X_opt[m] == 1] if X_opt[m].sum() else [" -"],
            )
        print(
            "先輩ドライバー\t:",
            *np.where(b_opt[y_opt == 1] == 1, "●", "×"),
            "\t",
            *self.simpl_date[b_opt == 1] if b_opt.sum() else [" -"],
        )

        print("--" * 64)
        print("訪問日別\n")
        visiters = {}
        for d in self.simpl_date[y_opt == 1]:
            visiters[d] = []
        for m, x in enumerate(X_opt):
            for d in range(len(self.simpl_date)):
                if X_opt[m, d] == 1:
                    visiters[self.simpl_date[d]].append(
                        (self.members[m] + "      ")[:6] + "\t"
                    )
        for d in range(len(self.candidate)):
            if y_opt[d] == 1 and b_opt[d] == 1:
                visiters[self.simpl_date[d]].append("先輩ドライバー\t")

        for d in visiters:
            print(f"{d}\t\t:", *visiters[d])

    def cons(self):
        self._cons_driver()
        self._cons_visiter()
        self._cons_visit_time()
        self._cons_senior()
        self._cons_project()


visit_model = VisitModel()
information, schedule, members, candidate, candidate_group, candidate_todo = read_data()
visit_model.verbose = 0
visit_model.add_vars(
    information, schedule, members, candidate, candidate_group, candidate_todo
)
visit_model.cons()
visit_model.set_objective()
visit_model.optimize()
visit_model.show_result()

delta = datetime.now() - start

print("==" * 64)
print(f"実行時間\t: {delta.seconds}.{delta.microseconds} 秒\n\n")
