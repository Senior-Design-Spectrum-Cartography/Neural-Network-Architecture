# Train the model
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=20, # Keep it low for preliminary testing
    callbacks=[
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    ]
)

# Evaluate on the unseen 20% test set
print("\nEvaluating on Test Set...")
test_loss, test_mae = model.evaluate(test_gen)
print(f"Test Loss (MSE): {test_loss:.4f}")
print(f"Test MAE (Mean Absolute Error in dB): {test_mae:.4f}")
