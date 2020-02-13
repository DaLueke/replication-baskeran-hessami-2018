
<a href="https://nbviewer.jupyter.org/github/HumanCapitalAnalysis/student-project-DaLueke/blob/master/student_project.ipynb"
   target="_parent">
   <img align="center" 
  src="https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.png" 
      width="109" height="20"> 
</a> 
<a href="https://mybinder.org/v2/gh/HumanCapitalAnalysis/student-project-DaLueke/master?filepath=student_project.ipynb"
    target="_parent">
    <img align="center"
       src="https://mybinder.org/badge_logo.svg"
       width="109" height="20">
</a>

# Student project: Replication of Baskeran & Hessami (2018)
This repository presents a replication of [Baskaran, T., & Hessami, Z. (2018)](https://www.aeaweb.org/articles?id=10.1257/pol.20170045). Does the election of a female leader clear the way for more women in politics? *American Economic Journal: Economic Policy*, 10(3), 95-121. The replication is accompanied by a critical review and some independend contributions towards the research question asked in the article.

## Abstract
In the context of the coursework for my Microeconometrics class, I present a replication of a collection of the results shown in Baskaran & Hessami (2018). Using a regression discontinuity design (RDD), the authors argue that they find a causal impact of female mayors on success of female council candidates. Throughout the replication, I can yield the same results for the baseline estimation model as well as the robustness tests that are presented in the original article. While most of the results regarding the internal validity of the RDD can successfully be replicated as well, I find indication that a local randomization might not be achieved. Even within a small bandwidth around the discontinuity, a relevant and significant explanatory power of municipalities' population sizes on success of council candidates can be found. 

## Contents of the repository
The coursework containing the replication as well as my own contributions can be found in the jupyter notebook *student_project.ipynb*. For better readability, a major part of the python code that is used throughout the replication was moved out of the notebook and can be found in the auxiliary subdirectory. Data for this replication is provided by the authors and available on the website of the [American Economic Association](https://www.aeaweb.org/doi/10.1257/pol.20170045.data). In order to execute the program codes provided with this repository, this data needs to be downloaded to the "data" subdirectory.

## References
**Baskaran, T., & Hessami, Z.** (2018) Does the Election of a Female Leader Clear the Way for More Women in Politics? *American Economic Journal: Economic Policy*, 10(3): 91-121

[![Build Status](https://travis-ci.org/HumanCapitalAnalysis/student-project-DaLueke.svg?branch=master)](https://travis-ci.org/HumanCapitalAnalysis/student-project-DaLueke) [![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](HumanCapitalAnalysis/student-project-template/blob/master/LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)
