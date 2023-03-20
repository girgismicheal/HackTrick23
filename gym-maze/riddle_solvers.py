import random
import time
import numpy as np
from PIL import Image
from amazoncaptcha import AmazonCaptcha
from jwcrypto import jwk
from jwcrypto import  jwt
import base64
import json
import base64
from scapy.all import *
from io import BytesIO
from scapy.layers.inet import IP
import re

generated_key= jwk.JWK.generate(kty='RSA', size=2048)
public_key= generated_key.export(private_key=False)


def cipher_solver(question):    
    # Add padding to the input string and decode from base64
    pad = question + "==" 
    decoded_ = base64.b64decode(pad)
    # Extract the shift and text values from the decoded string+
    old_text = decoded_[1:-1].split(b',')[0]
    # Convert the binary text to ASCII
    text = bytes(int(old_text[i:i+7], 2) for i in range(0, len(old_text), 7)).decode('ascii')
    # Compute the negative shift value
    shift = -int(decoded_[1:-1].split(b',')[1], 2)
    shifted_chars = [chr((ord(char) + shift - (97 - (32*char.isupper()))) % 26 + (97 - (32*char.isupper()))) for char in text]
    paint_text = ''.join(shifted_chars)
    # Return the shifted string
    return paint_text


def captcha_solver(question):
    image_array = np.array(question, dtype=np.uint8)
    image = Image.fromarray(image_array)
    captcha = AmazonCaptcha(image)
    return captcha.solve()


def server_solver(question):
    global generated_key, public_key

    # Return solution
    token_containt = question.split('.')
    header = json.loads(base64.b64decode(token_containt[0]+"==").decode('utf-8'))
    payload = json.loads(base64.b64decode(token_containt[1]+"==").decode('utf-8'))

    jwks = {"keys": [json.loads(public_key)]}
    header["jwk"] = jwks.get('keys')[0]
    header["kid"] = generated_key.thumbprint()
    payload['admin'] = 'true'
    
    # print(header)
    Token = jwt.JWT(header=header, claims=payload)
    Token.make_signed_token(generated_key)
    Token.validate(generated_key)
    return  Token.serialize()


def decode_base64(input_string): 
    # Add padding to the input string
    while len(input_string) % 4 != 0:
        input_string += "="
    # Decode the input string
    decoded_string = base64.b64decode(input_string).decode()
    return decoded_string


def pcap_solver(question):
  # Base64-encoded PCAP file
  base64_pcap = question

  # Decode the Base64 string
  pcap_data = base64.b64decode(base64_pcap)

  # Parse the binary data using scapy
  packets = rdpcap(BytesIO(pcap_data))

  # Set up the packet filter to show only packets for google.com
  domain_filter = "google.com"

  # query_names = set()
  decoded_query_components = set()
  # Process the packets as needed
  for packet in packets:
      # Filter for DNS packets with a query for the domain
      if packet.haslayer(DNSQR) and domain_filter in packet[DNSQR].qname.decode():
          # Extract the source and destination IP addresses and show the packet summary
          src_ip = packet[IP].src
          dst_ip = packet[IP].dst
          if src_ip or dst_ip == "188.68.45.12":
            query_name = packet[DNSQR].qname.decode()
            regex = r"\w+\.\w+\.google\.com"
            match = re.search(regex, query_name)
            if match:
              # query_names.add(query_name)
              index, part = tuple(query_name.split('.')[0:2])
              # print(index, part)
              decoded_query_components.add(
                  tuple(
                      [base64.b64decode(index + "==").decode(), base64.b64decode(part + "==").decode()]
                      )
                  )
  
  decoded_query_components = list(decoded_query_components)
  # print(decoded_query_components)
  # decoded_query_components.sort(key=lambda item: item[0])
  secret_key = ''.join([c[1] for c in sorted(decoded_query_components, key=lambda x:x[0])])
  return secret_key

