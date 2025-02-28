from tensorflow.keras import layers, models

class CNNModel:
    def __init__(self, input_shape, num_classes, metric):
        self.model = models.Sequential([
            layers.Input(shape=input_shape),
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.MaxPooling2D(2, 2),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D(2, 2),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D(2, 2),
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation='softmax')
        ])
    
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=metric)

    def train(self, train_data, validation_data, class_weight, epochs, batch_size):
        self.model.fit(train_data, validation_data=validation_data, class_weight=class_weight, epochs=epochs, batch_size=batch_size)

    def predict(self, x):
        y_pred = self.model.predict(x)
        return y_pred
    
    def save(self, path):
        self.model.save(path)
