# Amazon-Alexa-Reviews \  Machine leaning model 

### Table of Contents
- [Demo](#demo)
- [Overview](#overview)
- [Motivation](#motivation)
- [Technical Aspect](#technical-aspect)
- [Installation](#installation)
- [Run](#run)
- [Deployment on Render](#deployment-on-render)
- [Directory Tree](#directory-tree)
- [To Do](#to-do)
- [Bug / Feature Request](#bug--feature-request)
- [Technologies Used](#technologies-used)
- [Team](#team)
- [License](#license)
- [Credits](#credits)

---

## Demo
This project analyzes Amazon Alexa reviews to determine sentiments (positive, negative).<br>
**Link to Demo:** [Amazon-Alexa-Reviews Analysis](https://amazon-alexa-reviews.onrender.com) 

## Amazon-Alexa-Reviews

![Amazon Alexa Reviews Analysis](https://i.imgur.com/IhxPJfc.jpeg)


---

## Overview
The **Amazon-Alexa-Reviews** project leverages natural language processing (NLP) and machine learning techniques to analyze Amazon Alexa reviews. The goal is to classify the sentiment of the reviews to understand customer satisfaction and feedback trends.

Key features:
- Data preprocessing of Amazon Alexa reviews.
- Sentiment classification using machine learning models.
- Real-time predictions via a web application.

## Model Performance

### Evaluation Metrics
- **Accuracy**: 0.91
- **Precision**: 0.83
- **Recall**: 0.91
- **F1 Score**: 0.87

### Model Training Details
- **Classifier**: SVC (Support Vector Classifier)
- **Cross-Validation**: 3 folds for each of 10 candidates (totalling 30 fits)
- **Best Model**: SVC
- **Best F1 Score**: 0.90
---

## Motivation
Sentiment analysis of product reviews helps businesses:
- Understand customer feedback.
- Identify potential improvements for products.
- Monitor user sentiment on specific features or attributes.

This project showcases the application of NLP and machine learning for analyzing product reviews and classifying sentiments.


---

## Technical Aspect
### Training Machine Learning Models:
1. **Data Collection**:
   - Reviews are collected from Amazon's publicly available Alexa reviews dataset.

2. **Preprocessing**:
   - Cleaning the reviews by removing irrelevant text (URLs, special characters, etc.).
   - Tokenization, stop-word removal, and stemming/lemmatization.
   - Converting text to numerical representations using methods like TF-IDF or Word2Vec.

3. **Model Training**:
   - The project uses models like Logistic Regression, Naive Bayes, or more advanced transformers such as BERT.
   - Hyperparameter tuning and model evaluation using metrics like accuracy, precision, recall, and F1-score.

4. **Web Application**:
   - A Flask-based web app allows users to input reviews and get real-time sentiment predictions.
   - The app is deployed on Render for public access.
  
---

## Installation
The Code is written in Python 3.10. If you don't have Python installed you can find it [here](https://www.python.org/downloads/). If you are using a lower version of Python you can upgrade using the pip package, ensuring you have the latest version of pip. To install the required packages and libraries, run this command in the project directory after [cloning](https://www.howtogeek.com/451360/how-to-clone-a-github-repository/) the repository:

# To clone the repository

```bash

gh repo clone Creator-Turbo/Amazon-Alexa-Reviews

```
# Install dependencies: (all lib)
```bash
pip install -r requirements.txt
```



## Run
To train the Machine leaning models:
 To run the Flask web app locally
```bash
python app.py

```
# Deployment on Render

## To deploy the Flask web app on Render:
Deployment on Render

- To deploy the web app on Render:

- Push your code to GitHub.

- Log in to Render and create a new web service.

- Connect the GitHub repository.

- Configure environment variables (if any).

- Deploy and access your app live.


## Directory Tree 
```
Amazon-Alexa-Reviews/
│
├── data/                # Your data files
├── model/               # Your trained models
├── notebook/            # Jupyter notebooks
├── venv/                # Your virtual environment
├── static/              # Static files (CSS, JS, Images)
│   ├── css/
│   ├── js/
│   └── images/
├── templates/           # HTML templates
│   └── index.html
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
└── app.py               # Main Flask application
```

## To Do

- Expand dataset to improve model robustness.

- Experiment with advanced models like BERT or GPT-based sentiment classifiers.

- Add sentiment trend visualization to the web app.

- Automate data collection using the Twitter API.






## Bug / Feature Request
If you encounter any bugs or want to request a new feature, please open an issue on GitHub. We welcome contributions!




## Technologies Used
- Python 3.10

- scikit-learn

- Flask (for web app development)

- Render (for hosting and deployment)

- pandas (for data manipulation)

- numpy (for numerical computations)

- matplotlib (for visualizations)




![](https://forthebadge.com/images/badges/made-with-python.svg)


[<img target="_blank" src="https://upload.wikimedia.org/wikipedia/commons/thumb/0/05/Scikit_learn_logo_small.svg/260px-Scikit_learn_logo_small.svg.png" width=170>](https://pandas.pydata.org/docs/)
[<img target="_blank" src="https://miro.medium.com/v2/resize:fit:720/format:webp/0*RWkQ0Fziw792xa0S" width=170>](https://pandas.pydata.org/docs/)
  [<img target="_blank" src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSDzf1RMK1iHKjAswDiqbFB8f3by6mLO89eir-Q4LJioPuq9yOrhvpw2d3Ms1u8NLlzsMQ&usqp=CAU" width=280>](https://matplotlib.org/stable/index.html) 
 [<img target="_blank" src="https://icon2.cleanpng.com/20180829/okc/kisspng-flask-python-web-framework-representational-state-flask-stickker-1713946755581.webp" width=170>](https://flask.palletsprojects.com/en/stable/) 
 [<img target="_blank" src="https://upload.wikimedia.org/wikipedia/commons/thumb/3/31/NumPy_logo_2020.svg/512px-NumPy_logo_2020.svg.png" width=200>](https://aws.amazon.com/s3/) 
 [<img target="_blank" src="https://seaborn.pydata.org/_images/logo-mark-lightbg.svg" width=200>](https://seaborn.pydata.org/generated/seaborn.objects.Plot.html) 







## Team
This project was developed by:
[![Bablu kumar pandey](https://github.com/Creator-Turbo/images-/blob/main/resized_image.png?raw=true)](ressume_link) |
-|


**Bablu Kumar Pandey**


- [GitHub](https://github.com/Creator-Turbo)  
- [LinkedIn](https://www.linkedin.com/in/bablu-kumar-pandey-313764286/)
* **Personal Website**: [My Portfolio](https://creator-turbo.github.io/Creator-Turbo-Portfolio-website/)

## License

This project is licensed under the [MIT License](LICENSE).

You are free to use, modify, and distribute this software under the terms of the MIT License. For more details, see the [LICENSE](LICENSE) file included in this repository.


## Credits

Special thanks to the contributors of the scikit-learn library for their fantastic machine learning tools.
