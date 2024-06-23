from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load your pre-trained machine learning model from a saved pickle file
model = pickle.load(open('best_model.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def result():
    if request.method == 'POST':
        try:
            # Extract and convert user input data from the web form
            value_eur = float(request.form['value_eur'])
            age = int(request.form['age'])
            potential = int(request.form['potential'])
            movement_reactions = int(request.form['movement_reactions'])
            wage_eur = float(request.form['wage_eur'])
            mentality_composure = int(request.form['mentality_composure'])
            defending = int(request.form['defending'])
            dribbling = int(request.form['dribbling'])
            international_reputation = int(request.form['international_reputation'])
            skill_ball_control = int(request.form['skill_ball_control'])
            physic = int(request.form['physic'])
            attacking_crossing = int(request.form['attacking_crossing'])
            power_stamina = int(request.form['power_stamina'])

            # Create a list of feature values from the user input
            feature_values = [value_eur, age, potential, movement_reactions, wage_eur,
                              mentality_composure, defending, dribbling,international_reputation, skill_ball_control, physic,attacking_crossing,
                              power_stamina]

            # Make a prediction using the machine learning model
            to_predict = np.array(feature_values).reshape(1, -1)
            prediction = model.predict(to_predict)[0]

            # Average value of the Y_train (overall or rating) was 65.69907106564428
            average_target = 65.69907106564428
           
            # Calculate confidence as a percentage with 1 decimal place
            confidence = 100 * (1 - (abs(prediction - average_target) / average_target))


            return render_template("result.html", prediction=f'{prediction:.1f}', confidence = f'{confidence:.1f}')#add the confidence parameter and adjust the result.html to accept this

        except ValueError:
            return render_template("result.html", prediction="Invalid input values. Please enter valid numerical values.")

if __name__ == '__main__':
    app.run(debug=True)
