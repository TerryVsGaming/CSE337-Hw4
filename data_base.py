"""Calculate student grades by combining data from many sources.

Using Pandas, this script combines data from the:

* Roster
* Homework & Exam grades
* Quiz grades

to calculate final grades for a class.
"""
#Importing Libraries and Setting Paths
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats

HERE = Path(__file__).parent
DATA_FOLDER = HERE / "data"

#Data Importation and Cleaning

#ptB
roster = pd.read_csv(
    DATA_FOLDER/"roster.csv"
)
roster["Email Address"] = roster["Email Address"].apply(str.lower)
roster["NetID"] = roster["NetID"].apply(str.lower)
roster.set_index('NetID',inplace=True)


#ptC
hw_exam_grades = pd.read_csv(
    DATA_FOLDER/"hw_exam_grades.csv"
)
hw_exam_grades["SID"] = hw_exam_grades["SID"].apply(str.lower)
hw_exam_grades = hw_exam_grades[hw_exam_grades.columns.drop(list(hw_exam_grades.filter(regex='Submission')))]
hw_exam_grades.set_index('SID',inplace=True)


#ptD
quiz_grades = pd.DataFrame()

#Your code here to read the quiz_grades
for i in range(1,6):
    current_quiz = DATA_FOLDER/ f"quiz_{i}_grades.csv"
    quiz = pd.read_csv(current_quiz)
    quiz.rename(columns={'Grade': f'Quiz_{i}'}, inplace=True)
    quiz_grades = pd.concat([quiz_grades, quiz], ignore_index=True)

quiz_grades.fillna(0,inplace=True)
quiz_grades = quiz_grades.groupby(["Email", "Last Name", "First Name"]).sum().reset_index()
quiz_grades.set_index('Email', inplace=True)



#Pt3a Data Merging: roaster and homework
final_data = pd.merge(roster, hw_exam_grades, left_index=True, right_index=True)

# Merge with quiz grades using 'Email' as the common index
final_data = pd.merge(final_data, quiz_grades, left_on='Email Address', right_index=True)
final_data.drop(columns=['First Name_y', 'Last Name_y'], inplace=True)
final_data.rename(columns={'First Name_x': 'First Name', 'Last Name_x': 'Last Name'},inplace=True)

final_data = final_data.fillna(0)

# #Data Processing and Score Calculation
n_exams = 3

#For each exam, calculate the score as a proportion of the maximum points possible.
for n in range(1, n_exams + 1):
    final_data[f'Exam {n} Score'] = final_data[f'Exam {n}']/ final_data[f'Exam {n} - Max Points']


#Calculating Exam Scores:
#Filter homework and Homework - for max points
homework_scores = final_data.filter(regex='^Homework \d+$')
homework_max_points = final_data.filter(regex='^Homework \d+ - Max Points$')


# #Calculating Total Homework score
sum_of_hw_scores = homework_scores.sum(axis=1)
sum_of_hw_max = homework_max_points.sum(axis=1)
final_data["Total Homework"] = sum_of_hw_scores/sum_of_hw_max


# #Calculating Average Homework Scores
# hw_max_renamed =
average_hw_scores = np.mean(homework_scores.values/homework_max_points.values, axis=1)
final_data["Average Homework"] = average_hw_scores

# #Final Homework Score Calculation
final_data["Homework Score"] = np.maximum(final_data["Total Homework"].values, final_data["Average Homework"].values)


# #Calculating Total and Average Quiz Scores:
# #Filter the data for Quiz scores
quiz_scores = final_data.filter(regex='Quiz')

#
quiz_max_points = pd.Series(
    {"Quiz 1": 11, "Quiz 2": 15, "Quiz 3": 17, "Quiz 4": 14, "Quiz 5": 12}
)

# #Final Quiz Score Calculation:
sum_of_quiz_scores = quiz_scores.sum(axis=1)
sum_of_quiz_max = quiz_max_points.sum()
final_data["Total Quizzes"] = sum_of_quiz_scores/sum_of_quiz_max

# Average Quiz Scores
average_quiz_scores = np.mean(quiz_scores.values/quiz_max_points.values, axis=1)
final_data["Average Quizzes"] = average_quiz_scores

#
final_data["Quiz Score"] = np.maximum(final_data["Total Quizzes"].values, final_data["Average Quizzes"].values)

#Calculating the Final Score:
weightings = pd.Series(
    {
        "Exam 1 Score": 0.05,
        "Exam 2 Score": 0.1,
        "Exam 3 Score": 0.15,
        "Quiz Score": 0.30,
        "Homework Score": 0.4,
    }
)
temp_df = final_data[["Exam 1 Score", "Exam 2 Score", "Exam 3 Score", "Quiz Score", "Homework Score"]]
final_data["Final Score"] = np.matmul(temp_df.values, weightings.values.reshape(-1, 1))

#Rounding Up the Final Score:
final_data["Ceiling Score"] = np.ceil(final_data["Final Score"]*100)


#Defining Grade Mapping:
grades = {
    90: "A",
    80: "B",
    70: "C",
    60: "D",
    0: "F",
}

#Applying Grade Mapping to Data:
def grade_mapping(value):
    for threshold, grade in sorted(grades.items(), reverse=True):
        if value >= threshold:
            return grade
    return "F"


# letter_grades =
final_data["Final Grade"] = final_data["Ceiling Score"].apply(lambda x: grade_mapping(x))


# #Processing Data by Sections:
for section, table in final_data.groupby("Section"):
    table.sort_values(by=["First Name", "Last Name"], inplace=True)
    num_students = table.shape[0]
    file_name = f"Sorted Section {section} Grades.csv"
    table.to_csv(DATA_FOLDER/file_name)
    print(f"In Section {section}, there are {num_students} saved to file Section {DATA_FOLDER/file_name}")

# #Visualizing Grade Distribution: Get Grade Counts and use plot to plot the grades
grade_counts = final_data["Ceiling Score"].apply(lambda x: grade_mapping(x)).value_counts().reindex(grades.values(), fill_value=0)

#Visualize the data on with Histogram and use Matplot lib density function to print Kernel Density Estimate
final_data["Final Score"].plot.hist(bins=20, label="Histogram")
final_data["Final Score"].plot.density(
    linewidth=4, label="Kernel Density Estimate"
)




#Plotting Normal Distribution:
final_mean = final_data["Final Score"].mean()
final_std = final_data["Final Score"].std()

#Plot the normal distribution of final_mean and final_std
sample_x = np.linspace(final_mean-final_std*5, final_mean+final_std*5, 1000)
pdf_values = scipy.stats.norm.pdf(sample_x, final_mean, final_std)
plt.plot(sample_x, pdf_values,label="Normal Distribution", color="green")
plt.legend()
plt.show()