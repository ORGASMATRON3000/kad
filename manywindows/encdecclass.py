from Cryptodome.PublicKey import RSA
from Cryptodome.Cipher import PKCS1_OAEP
from Cryptodome.Signature import PKCS1_v1_5
from Cryptodome.Hash import SHA512, SHA384, SHA256, SHA, MD5
from Cryptodome import Random
from base64 import b64encode, b64decode
hash = "SHA-512"

def newkeys(keysize):
    random_generator = Random.new().read
    key = RSA.generate(keysize, random_generator)
    private, public = key, key.publickey()
    return public, private
def decrypt(ciphertext, priv_key):
    #RSA encryption protocol according to PKCS#1 OAEP
    cipher = PKCS1_OAEP.new(priv_key)
    return cipher.decrypt(ciphertext)

def sign(message, priv_key, hashAlg="SHA-256"):
    global hash
    hash = hashAlg
    print(hash)
    signer = PKCS1_v1_5.new(priv_key)
    if (hash == "SHA-512"):
        digest = SHA512.new()
    elif (hash == "SHA-384"):
        digest = SHA384.new()
    elif (hash == "SHA-256"):
        digest = SHA256.new()
    elif (hash == "SHA-1"):
        digest = SHA.new()
    else:
        digest = MD5.new()
    digest.update(message)
    return signer.sign(digest)

def verify(message, signature, pub_key):
    signer = PKCS1_v1_5.new(pub_key)
    print(hash)
    if (hash == "SHA-512"):
        digest = SHA512.new()
    elif (hash == "SHA-384"):
        digest = SHA384.new()
    elif (hash == "SHA-256"):
        digest = SHA256.new()
    elif (hash == "SHA-1"):
        digest = SHA.new()
    else:
        digest = MD5.new()
    digest.update(message)
    
    return signer.verify(digest, signature)
#Create and save keys

def createKeys(companyName):
    public,private=newkeys(4096)
    f = open(companyName+"_public.pem", 'wb')
    f.write(public.exportKey('PEM'))
    f.close()
    f = open(companyName+"_private.pem", 'wb')
    f.write(private.exportKey('PEM'))

def signMessage(companyName,message):
    f = open(companyName+"_private.pem", 'rb')
    private = RSA.importKey(f.read())
    f.close()
    signature = b64encode(sign(message.encode('utf-8'), private, "SHA-512"))
    return (signature)
def verifyMessage(companyName,message,signature):
    f = open(companyName+"_public.pem", 'rb')
    public = RSA.importKey(f.read())
    f.close()
    
    return verify(message.encode('utf-8'), b64decode(signature), public)
