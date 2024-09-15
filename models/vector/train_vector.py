import math
import time
import torch
from torch_geometric.loader import DataLoader

def train_vector_model(model, device, optimizer, criterion, n_epochs, batch_size, validate, training_set, validation_set, shuffle_training_set, shuffle_validation_set):
	# creating DataLoaders
	training_loader = DataLoader(dataset=training_set, batch_size=batch_size, shuffle=shuffle_training_set)
	if validate:
		validation_loader = DataLoader(dataset=validation_set, batch_size=batch_size, shuffle=shuffle_validation_set)

	total_samples = len(training_set)
	n_iterations = math.ceil(total_samples/batch_size)
	    
	# saving the losses from every epoch
	training_losses = []
	validation_losses = []

	start_time = time.perf_counter()
	    
	for epoch in range(n_epochs):
		# tracking loss per epoch
		training_running_loss = 0.0
		n_training_batches = 0
		validation_running_loss = 0.0
		n_validation_batches = 0

		# training phase
		model.train()

		for i, graph in enumerate(training_loader):
			n_training_batches += 1

			# operators applied to the focal building
			operators = graph.y
	        
			# moving the features to device
			graph = graph.to(device)
			operators = operators.to(device)
	    
			# empty the gradients
			optimizer.zero_grad()
	            
			# forward pass
			pred_operators_logits = model(graph.x_dict, graph.edge_index_dict) # compute predictions, calls forward method under the hood
			loss = criterion(pred_operators_logits, operators) # calculate loss
			training_running_loss += loss.item() # tracking running loss to keep track of the loss for every epoch
	    
			# backward pass
			loss.backward() # backpropagation
			optimizer.step() # update the parameters
	    
			# print information every few batches
			if not (i + 1) % (n_iterations // 10):
				print(f"epoch {epoch+1}/{n_epochs}, step {i+1}/{n_iterations}")

		training_loss_epoch = training_running_loss / n_training_batches
		training_losses.append(training_loss_epoch)

		if validate:
			# validation phase
			model.eval()

			with torch.no_grad():
				for graph in validation_loader:
					n_validation_batches += 1

					# operators applied to the focal building
					operators = graph.y
		            
					# moving the features to device
					graph = graph.to(device)
					operators = operators.to(device)

					# prediction on the trained model results in logits
					pred_operators_logits = model(graph.x_dict, graph.edge_index_dict) # compute predictions, calls forward method under the hood
					# calculate and store validation loss
					loss = criterion(pred_operators_logits, operators)
					validation_running_loss += loss.item()

			validation_loss_epoch = validation_running_loss / n_validation_batches
			validation_losses.append(validation_loss_epoch)
	    
			print(f"epoch {epoch+1} finished, training loss: {training_loss_epoch:.3f}, validation loss: {validation_loss_epoch:.3f}")
		else:
			validation_losses.append(0)
			print(f"epoch {epoch+1} finished, training loss: {training_loss_epoch:.3f}")

	end_time = time.perf_counter()
	training_time = end_time - start_time
	print(f"Training time: {training_time:,.3f} seconds")

	return model, training_losses, validation_losses, training_time