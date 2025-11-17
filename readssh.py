import paramiko

hostname = '10.5.1.116'
username = 'jonathanyeh'
password = 'Glt4262828@' # Or use key_filename for key-based authentication
# key_filename = '/path/to/your/private_key' 

try:
    # Create an SSH client
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy()) # Automatically add the server's host key

    # Connect to the remote server
    if password:
        client.connect(hostname=hostname, username=username, password=password)
    # elif key_filename:
    #     client.connect(hostname=hostname, username=username, key_filename=key_filename)
    else:
        print("Error: Please provide either a password or a private key file.")
        exit()

    # Execute the nvidia-smi command
    stdin, stdout, stderr = client.exec_command('nvidia-smi') #watch -n 1 nvidia-smi 

    # Read the output
    output = stdout.read().decode('utf-8')
    error = stderr.read().decode('utf-8')

    if output:
        print("nvidia-smi output:")
        print(output)
    if error:
        print("nvidia-smi error:")
        print(error)

except paramiko.AuthenticationException:
    print("Authentication failed. Check your username and password/key.")
except paramiko.SSHException as e:
    print(f"SSH connection error: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
finally:
    # Close the SSH connection
    if 'client' in locals() and client:
        client.close()
