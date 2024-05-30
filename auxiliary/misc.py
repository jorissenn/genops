import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

def send_notification(model, email_address, n_samples, n_epochs, batch_size):
    '''Sends an email from the cluster to inform once the training process of a given model has finished.'''
    
    FROM = "noreply-s3it@zi.uzh.ch"
    TO = email_address
    SUBJECT = "Training finished!"
    TEXT = f"The training process of {model} has finished ({n_samples:,} samples over {n_epochs:,} epochs with batch size {batch_size:,})."
    
    # Create a multipart message and set headers
    message = MIMEMultipart()
    message['From'] = FROM
    message['To'] = TO
    message['Subject'] = SUBJECT
    
    # Add body to the email
    message.attach(MIMEText(TEXT, 'plain'))
    
    # Send the email
    server = smtplib.SMTP('localhost')
    server.sendmail(FROM, TO, message.as_string())
    server.quit()
    