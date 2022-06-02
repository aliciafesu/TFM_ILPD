from flask import Flask, request, render_template
import pandas as pd
import joblib


# Crear la aplicación
app = Flask(__name__)

# Funcion principal
@app.route('/', methods=['GET', 'POST'])
def main():
    
    # request
    if request.method == "POST":
        
        # llamar al modelo
        gnb = joblib.load("gnb.pkl")
        
        # tomar los valores indicados
        Age = request.form.get("Age")
        Gender = request.form.get("Gender")
        TB = request.form.get("TB")
        DB = request.form.get("DB")
        Alkphos = request.form.get("Alkphos")
        Sgpt = request.form.get("Sgpt")
        ALB = request.form.get("ALB")
        AG = request.form.get("AG")
        
        # crear un dataframe con las entradas
        X = pd.DataFrame([[Age, Gender, TB, DB, Alkphos, Sgpt, ALB, AG]], columns = ["Age", "Gender","TB","DB","Alkphos","Sgpt","ALB","AG"])
        
        # predicción
        prediction = gnb.predict(X)[0]
        
    else:
        prediction = ""
        
    return render_template("app_test1.html", Resultado = prediction)

# correr la app 
if __name__ == '__main__':
    app.run(debug = True)

