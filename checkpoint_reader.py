'''
Created on 2017年11月1日

@author: zry
'''
import tensorflow as tf
import sys
 
if __name__ == '__main__':
#   sys.stdout = open(r'C:\Users\zry\Desktop\cef.txt','w')
  linears = [
      "para_to_reg",
      "input_to_reg",
      "para_to_mem_addr",
      "mem_to_reg",
      "para_to_reg_id",
      "reg_to_reg_add",
      "reg_to_mem",
      "para_to_ins_addr",
      "reg_to_prob_jump",
      "reg_to_output",
    ]
  reader = tf.train.NewCheckpointReader(r'C:\Users\zry\Desktop\checkpoint\model.ckpt-340000')
  print(reader.get_tensor("instructions"))
  print(reader.get_tensor("parameters"))
#   for tensor_name in linears:
#     tensor_name_w = "AI_programming/"+tensor_name+"/w"
#     print(tensor_name_w)
#     print(reader.get_tensor(tensor_name_w))
#     if not tensor_name == "reg_to_output":
#       tensor_name_b = "AI_programming/"+tensor_name+"/b"
#       print(tensor_name_b)
#       print(reader.get_tensor(tensor_name_b))
  
  
  