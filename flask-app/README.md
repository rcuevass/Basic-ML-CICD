# Flask app

Folder reserved for flask app

#### Notes

1. To set development environment, as opposed to production, do
`export FLASK_ENV=development` and then `flask run` to execute 
application.

2. To test app flask POST:

  - `curl http://localhost:5000`
  - `curl -F "mnist_image=@/Users/rcuevas/PycharmProjects/Basic-ML-CICD/data/test/image_10.jpg" http://localhost:5000/predict`