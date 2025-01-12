import numpy as np
import pandas as pd
import os
import cv2


class ComparisonModule:
    def __init__(self, name="ComparisonModule"):
        self.name = name

    def calculate_similarity(self, x: np.ndarray, y: np.ndarray) -> float:
        raise NotImplementedError("calculate_similarity not implemented")


def generate_csv(
    csv_path: str,
    modules: list[ComparisonModule],
    data_path: str = "data/clean",
    verbose: bool = False,
    min_max_similarities=False,
) -> None:
    """
    Generates a csv file with similarity between every final and inspiration image for all groups
    """
    groups = sorted(os.listdir(data_path))
    final_df = pd.DataFrame(
        columns=["Final_Submission", "Inspiration"]
        + [f"{x.name}_Similarity" for x in modules]
        + ["Source"]
    )
    for group in groups:
        if verbose:
            print(f"Processing group {group}")
        group_path = os.path.join(data_path, group)

        final = [
            os.path.join(group_path, "final", x)
            for x in os.listdir(os.path.join(group_path, "final"))
        ]
        web = [
            os.path.join(group_path, "web", x)
            for x in os.listdir(os.path.join(group_path, "web"))
        ]
        ai = [
            os.path.join(group_path, "ai", x)
            for x in os.listdir(os.path.join(group_path, "ai"))
        ]

        group_dict = {x: [] for x in final_df.columns}

        for f in final:
            for w in web:
                f_name = f"{group}_" + f.split("/")[-1]
                w_name = w.split("/")[-1]

                f_cv2 = cv2.imread(f)
                w_cv2 = cv2.imread(w)

                group_dict["Final_Submission"].append(f_name)
                group_dict["Inspiration"].append(w_name)
                group_dict["Source"].append("web")
                for x in modules:
                    group_dict[f"{x.name}_Similarity"].append(
                        x.calculate_similarity(f_cv2, w_cv2)
                    )

            for a in ai:
                f_name = f"{group}_" + f.split("/")[-1]
                a_name = a.split("/")[-1]

                f_cv2 = cv2.imread(f)
                a_cv2 = cv2.imread(a)

                group_dict["Final_Submission"].append(f_name)
                group_dict["Inspiration"].append(a_name)
                group_dict["Source"].append("ai")
                for x in modules:
                    group_dict[f"{x.name}_Similarity"].append(
                        x.calculate_similarity(f_cv2, a_cv2)
                    )

        final_df = (
            pd.concat([final_df, pd.DataFrame(group_dict)], ignore_index=True)
            if final_df.size > 0
            else pd.DataFrame(group_dict)
        )

    if min_max_similarities:
        for x in modules:
            series = final_df[f"{x.name}_Similarity"]
            final_df[f"{x.name}_Similarity"] = (series - series.min()) / (
                series.max() - series.min()
            )

    final_df.to_csv(csv_path, index=False)


__all__ = ["ComparisonModule", "generate_csv"]
