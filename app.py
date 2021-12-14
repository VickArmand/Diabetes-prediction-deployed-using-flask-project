from flask import Flask,render_template,request
import pickle
import numpy as np
app=Flask(__name__)
model =pickle.load(open("diabetespredictionmodel.pkl","rb"))

@app.route('/')
def index():
    return render_template("diabetespred.html")
@app.route('/predictions',methods=['POST'])
def predict():
    if request.method=='POST':
        # pregnanciesno=request.form["pregnanciesno"]
        # glucose=request.form["glucose"]
        # bmi=request.form["bmi"]
        # insulin=request.form["insulin"]
        # Age=request.form["Age"]
        # bloodpressure=request.form["bloodpressure"]
        # outcome=request.form["outcome"]
        # DiabetesPedigreeFunction=request.form["DiabetesPedigreeFunction"]
        # SkinThickness=request.form["SkinThickness"]
        features=[float(x) for x in request.form.values()]
        finalfeatures=[np.array(features)]
        prediction=model.predict(finalfeatures)
        if prediction==1:

          return render_template("diabetespred.html",pred="Higher chances of diabetes")
        if prediction==0:

          return render_template("diabetespred.html",pred="Minimal chances of diabetes")


if __name__ == "__main__":
    app.run(debug=True)
