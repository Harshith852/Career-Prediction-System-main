# Career Path Predictor AI 🎯

An intelligent career prediction system that uses machine learning to suggest optimal career paths based on individual skills, preferences, and characteristics.

## 🌟 Features

- **AI-Powered Predictions**: Utilizes multiple ML models including Decision Trees, SVM, Random Forest, and XGBoost
- **Interactive Web Interface**: Built with Streamlit for a user-friendly experience
- **Real-time Analytics**: Dynamic visualizations and insights about career trends
- **Multi-model Ensemble**: Combines predictions from multiple models for better accuracy
- **Comprehensive Assessment**: Evaluates technical skills, soft skills, and personal preferences
- **Data Persistence**: Stores user data and predictions in a database
- **Visual Analytics**: Interactive charts and graphs for better understanding

## 🛠️ Technology Stack

- **Backend**: Python 3.x
- **Frontend**: Streamlit, HTML/CSS
- **Machine Learning**: scikit-learn, XGBoost
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly, Matplotlib, Seaborn
- **Database**: SQLite
- **GUI Alternative**: Tkinter (separate interface available)

## 📋 Prerequisites

- Python 3.x
- pip package manager
- Virtual environment (recommended)

## 🚀 Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd career-prediction-system
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. Prepare the data:
```bash
# Create necessary directories
mkdir -p data pkl
```

## 💻 Usage

1. Train the models:
```bash
python pythonFunctions/training.py
```

2. Run the web application:
```bash
streamlit run app.py
```

3. Alternative GUI interface:
```bash
python pythonFunctions/GUI.py
```

## 📁 Project Structure

```
career-prediction-system/
├── app.py                    # Main Streamlit web application
├── pythonFunctions/
│   ├── training.py          # Model training script
│   └── GUI.py               # Alternative Tkinter GUI
├── data/
│   └── mldata.csv           # Dataset (not included in repo)
├── pkl/                     # Directory for trained models
├── db.py                    # Database operations
└── requirements.txt         # Project dependencies
```

## 🔍 Features Evaluated

- Logical Quotient Rating
- Coding Skills
- Hackathons
- Public Speaking
- Self-Learning Capability
- Extra Courses
- Certifications
- Workshops
- Team Work Experience
- Technical vs Management Interests
- And more...

## 📊 Model Performance

The system uses an ensemble of multiple models:
- Decision Tree Classifier
- Support Vector Machine (SVM)
- Random Forest Classifier
- XGBoost Classifier

Average accuracy across models: ~85-90%

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📚 References

This project is built upon research from various academic papers and studies in the field of career prediction and machine learning. For detailed references, see `references.bib`.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ✨ Acknowledgments

- Research papers and studies cited in `references.bib`
- Contributors and maintainers
- Open source community

## 👤 Author

Harshith
- GitHub: [@harshith852](https://github.com/harshith852/)
