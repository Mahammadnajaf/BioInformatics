import os


def read_scores_file(filename):
    scores = {}
    with open(filename) as f:
        for line in f:
            if line.startswith("Binary Files:"):
                # extract the filenames and score
                fields = line.split(", ")
                file1 = fields[0].split("#")[-1].replace(".npy", "")
                file2 = fields[1].split("#")[-1].replace(".npy", "")
                score = float(fields[2].split(": ")[-1])

                # add the score to the dictionary for both files
                if file1 not in scores:
                    scores[file1] = []
                scores[file1].append(score)

                if file2 not in scores:
                    scores[file2] = []
                scores[file2].append(score)
    return scores


def get_imposter_performance(scores):
    return {key: sum(score_list) / len(score_list) for key, score_list in scores.items()}


def get_worst_performing_subject(filename, other_files):
    # read the scores for the main file
    scores = read_scores_file(filename)

    # get the imposter performances for the main file
    imposter_performances = get_imposter_performance(scores)

    # get the worst-performing subjects from the main file
    worst_subjects = sorted(imposter_performances.items(), key=lambda x: x[1], reverse=True)[:5]
    print("Worst Performing Subjects in", filename + ":")
    for subject, avg_score in worst_subjects:
        rank = sorted(imposter_performances.values(), reverse=True).index(avg_score) + 1
        print(f"{subject}: rank {rank}, average score {avg_score}")

    # check if the worst-performing subjects from the main file exist in other files
    common_subjects = set(imposter_performances.keys())
    for other_file in other_files:
        other_scores = read_scores_file(other_file)
        common_subjects.intersection_update(other_scores.keys())

    # print the rankings for the common subjects in the main file
    for subject in sorted(common_subjects):
        rank = sorted(scores[subject], reverse=True).index(imposter_performances[subject]) + 1
        print(f"{subject} in {filename}: rank {rank}")


get_worst_performing_subject('a16_imp_9march.txt', ['ve_imp_9march.txt', 'd16_imp_9march.txt'])
