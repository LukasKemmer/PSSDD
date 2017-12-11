import paramiko as pa
import sys
import os

USER_NAME = "dsluser32"

def connect():
    client = pa.SSHClient()
    client.set_missing_host_key_policy(pa.AutoAddPolicy())
    client.connect(hostname="40.114.149.233", username=USER_NAME, password="twXE5s")
    return client


def upload(sclient, localpath, remotepath):
    fclient = sclient.open_sftp()
    fclient.put(localpath, remotepath)
    fclient.close()


def execute(sclient, remotepath):
    stdin,stdout,stderr = sclient.exec_command("/anaconda/bin/python " + remotepath, get_pty=True)
    stdin.close()
    for line in iter(lambda: stdout.readline(2048), ""):
        print(line, end="")


def main():
    localpath = sys.argv[1]
    remotepath = "/home/" + USER_NAME + "/kaggle/src/"+os.path.basename(os.path.normpath(localpath))

    print("Connecting...")
    sclient = connect()

    print("Uploading " + localpath + " to " + remotepath)
    upload(sclient, localpath, remotepath)

    print("Executing...")
    execute(sclient, remotepath)


main()