class FacePredictor(object):
    def __init__(self, face_preprocessor, face_classifier):
        self.face_preprocessor = face_preprocessor
        self.face_classifier = face_classifier
    
    def predict_proba(self, face):
        face_prep = self.face_preprocessor.transform(face)
        face_prob = self.face_classifier.predict(face_prep)
        return face_prob