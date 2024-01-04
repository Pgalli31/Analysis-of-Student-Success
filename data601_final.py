import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


data = pd.read_csv('final_edited_data.csv')

Y = data['num_grade']
x = data[['weekly_study_hours', 'reading_frequency', 'seminar_attendance',
          'class_attendance', 'study_when', 'taking_notes', 'age', 'sex', 'high_school_type', 'scholarship_type',
          'additional_work', 'additional_activity', 'partner_status', 'salary', 'transportation_method', 'housing',
          'mother_education', 'father_education', 'siblings', 'parental_status', 'mother_occupation', 'father_occupation',
          'academic_reading_frequency', 'project_impact', 'study_with', 'listening_in_class', 'discussion',
          'flip_classroom']]


X = pd.get_dummies(x, columns=['weekly_study_hours', 'reading_frequency', 'seminar_attendance',
                               'class_attendance', 'study_when', 'taking_notes', 'age', 'sex', 'high_school_type', 'scholarship_type',
                               'additional_work', 'additional_activity', 'partner_status', 'salary', 'transportation_method', 'housing',
                               'mother_education', 'father_education', 'siblings', 'parental_status', 'mother_occupation', 'father_occupation',
                               'academic_reading_frequency', 'project_impact', 'study_with', 'listening_in_class', 'discussion',
                               'flip_classroom'], drop_first=True)

X.drop('father_education_high school', axis=1, inplace=True)
X.drop('high_school_type_state', axis=1, inplace=True)
X.drop('mother_occupation_housewife', axis=1, inplace=True)
X.drop('transportation_method_bus', axis=1, inplace=True)
X.drop('study_when_never', axis=1, inplace=True)
X.drop('discussion_never', axis=1, inplace=True)
X.drop('scholarship_type_none', axis=1, inplace=True)
X.drop('mother_occupation_private sector employee', axis=1, inplace=True)
X.drop('reading_frequency_sometimes', axis=1, inplace=True)
X.drop('partner_status_yes', axis=1, inplace=True)
X.drop('siblings_5', axis=1, inplace=True)
X.drop('father_education_masters', axis=1, inplace=True)
X.drop('father_occupation_private sector employee', axis=1, inplace=True)
X.drop('class_attendance_sometimes', axis=1, inplace=True)
X.drop('father_occupation_other', axis=1, inplace=True)
X.drop('age_22-25', axis=1, inplace=True)
X.drop('housing_with family', axis=1, inplace=True)
X.drop('academic_reading_frequency_sometimes', axis=1, inplace=True)
X.drop('academic_reading_frequency_often', axis=1, inplace=True)
X.drop('taking_notes_sometimes', axis=1, inplace=True)
X.drop('listening_in_class_never', axis=1, inplace=True)
X.drop('project_impact_positive', axis=1, inplace=True)
X.drop('study_when_regularly throughout semester', axis=1, inplace=True)
X.drop('listening_in_class_sometimes', axis=1, inplace=True)
X.drop('additional_activity_yes', axis=1, inplace=True)
X.drop('scholarship_type_75%', axis=1, inplace=True)
X.drop('father_occupation_self-employed', axis=1, inplace=True)
X.drop('housing_other', axis=1, inplace=True)
X.drop('study_with_with friends', axis=1, inplace=True)
X.drop('discussion_sometimes', axis=1, inplace=True)
X.drop('project_impact_neutral', axis=1, inplace=True)
X.drop('flip_classroom_useful', axis=1, inplace=True)
X.drop('taking_notes_never', axis=1, inplace=True)
X.drop('transportation_method_private car/taxi', axis=1, inplace=True)
X.drop('father_education_university', axis=1, inplace=True)
X.drop('father_education_secondary school', axis=1, inplace=True)
X.drop('weekly_study_hours_6-10 hours', axis=1, inplace=True)
X.drop('weekly_study_hours_none', axis=1, inplace=True)
X.drop('weekly_study_hours_<5 hours', axis=1, inplace=True)
X.drop('salary_271-340', axis=1, inplace=True)
X.drop('father_education_primary school', axis=1, inplace=True)
X.drop('siblings_3', axis=1, inplace=True)
X.drop('salary_341-410', axis=1, inplace=True)
X.drop('salary_201-270', axis=1, inplace=True)
X.drop('siblings_4', axis=1, inplace=True)
X.drop('mother_occupation_retired', axis=1, inplace=True)
X.drop('mother_education_university', axis=1, inplace=True)
X.drop('seminar_attendance_yes', axis=1, inplace=True)
X.drop('siblings_2', axis=1, inplace=True)
X.drop('mother_education_secondary school', axis=1, inplace=True)
X.drop('mother_education_high school', axis=1, inplace=True)
X.drop('mother_education_primary school', axis=1, inplace=True)
X.drop('salary_above 410', axis=1, inplace=True)
X.drop('father_occupation_retired', axis=1, inplace=True)
X.drop('age_above 26', axis=1, inplace=True)
X.drop('sex_male', axis=1, inplace=True)


X = sm.add_constant(X.astype(int))

ks = sm.OLS(Y, X)
ks_res = ks.fit()
ks_res.summary()


keep_pvals = ks_res.pvalues[ks_res.pvalues >= .05]

