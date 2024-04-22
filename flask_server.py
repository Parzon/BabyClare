from flask import Flask
app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello, World!'

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port = 5001)

'''''
The first line of code `from flask import Flask` is basically importing Flask package in the file. 
We next create an object of the flask class using `app = flask()`
We send the following argument to the default constructor `__name__`, this will help Flask look for templates and static files. 
Next, we use the route decorator to help define which routes should be navigated to the following function. `@app.route(/)`
In this, when we use `/` it lets Flask direct all the default traffic to the following function. 
We define the function and the action to be performed. 
The default content type is HTML, so our string will be wrapped in HTML and displayed in the browser. 
In the main function created, `app.run()` will initiate the server to run on the local development server. 

'''''