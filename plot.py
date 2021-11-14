import matplotlib.pyplot as plt
seed = 73
pth = f'50cnn_log{seed}.txt'
X1=[]
Y1=[]
X2=[]
Y2=[]
X3=[]
Y3=[]
X4=[]
Y4=[]
sum_b=0
sum_agent2 = 0
# seed_set = set([0,2,4,13,18,25,26,28,42,53,73,82,102,114,115,132,139,150,154,168,172,173,186])
seed_set = [4,13,73,82,122,124,139,150,154,173,186]
for seed in seed_set:
    pth = f'log/40_{seed}cnn_log.txt'
    # pth = f'50cnn_log{seed}.txt'
    with open(pth,'r') as f:
        lines = f.readlines()
        # print(len(lines))
        for line in lines:
            val = [float(s) for s in line.split()] # 5, v1,b1,v2,b2,gt
            if val[1]>val[3]:
                sum_b += val[0]<val[2]
                sum_agent2 += 1
            # if val[0] > val[2]: # b1>b2,v1>v2
            #     X1.append(val[1])
            #     Y1.append(val[3])
            # elif val[0] < val[2]: # b1>b2,v1<v2
            #     X2.append(val[1])
            #     Y2.append(val[3])    
            X1.append(abs(val[1]-val[3]))
            Y1.append(abs(val[0]-val[2]))
            # X2.append(val[1])
            # Y2.append(val[0])
    # break
plt.plot([0,0.1,1],[0,0.1,1])
print(sum_b, sum_agent2)
plt.scatter(X1,Y1,alpha=0.3,label='v1>v2')
# plt.scatter(X2,Y2,alpha=0.3,label='v1<v2')
# plt.scatter(X3,Y3,alpha=0.3,label='b1<b2,v1>v2')
# plt.scatter(X4,Y4,alpha=0.3,label='b1<b2,v1<v2')
plt.xlabel('Bid differnce')
plt.ylabel('Value difference')
plt.title(f'Training')
# plt.legend()
# plt.axis([0.0, 0.002, 0.0, 5])
plt.show()

# for i in range(1,50):
#     plt.subplot(7,7,i)
#     pth = f'./log/{i}_124cnn_log.txt'
#     X1=[]
#     Y1=[]
#     X2=[]
#     Y2=[]
#     sum_b=0
#     sum_agent2 = 0
#     with open(pth,'r') as f:
#         lines = f.readlines()
#         for line in lines:
#             val = [float(s) for s in line.split()] # 5, v1,b1,v2,b2,gt
#             if val[1]>val[3]:
#                 # print(val[1],val[3])
#                 sum_b += val[0]<val[2]
#                 sum_agent2 += 1
                
#             X1.append(val[0])
#             Y1.append(val[1])
#             X2.append(val[2])
#             Y2.append(val[3])
#     print(sum_b, sum_agent2)
#     print(len(X1))
#     plt.scatter(X1,Y1,alpha=0.6,label='agent1')
#     plt.scatter(X2,Y2,alpha=0.6,label='agent2')
#     plt.xlabel(f'Evaluation{i}')
#     plt.ylabel('Bid')
#     plt.legend()
# plt.show()

# pth = '50cnn_log.txt'
# X1=[]
# Y1=[]
# X2=[]
# Y2=[]
# sum_b=0
# sum_agent2 = 0
# with open(pth,'r') as f:
#     lines = f.readlines()
#     for line in lines:
#         val = [float(s) for s in line.split()] # 5, v1,b1,v2,b2,gt
#         if val[1]>val[3]:
#             # print(val[1],val[3])
#             sum_b += val[0]<val[2]
#             sum_agent2 += 1
            
#         X1.append(val[0])
#         Y1.append(val[1])
#         X2.append(val[2])
#         Y2.append(val[3])
# print(sum_b, sum_agent2)
# print(len(X1))
# plt.scatter(X1,Y1,alpha=0.6,label='agent1')
# plt.scatter(X2,Y2,alpha=0.6,label='agent2')
# plt.xlabel(f'Evaluation{0}')
# plt.ylabel('Bid')
# plt.legend()
# plt.show()