import numpy as np
import matplotlib.pyplot as plt
import pickle



# # plant
# loss2 = plaint_line(loss_contents1)
# fig,ax1 = plt.subplots(figsize=(7.5,4.8))
# ax2 = ax1.twinx()
# lns1 = ax1.plot(np.arange(len(loss1)),loss1,label = "content_loss",linewidth=1)
# lns2 = ax2.plot(np.arange(len(loss_contents)), loss_contents, label="y_loss",color="red",linewidth=1)
# ax1.set_xlabel('iteration')
# ax1.set_ylabel('content_loss')
# ax2.set_ylabel('y_loss')
# lns = lns1 + lns2
# labels = ["content_loss","y_loss"]
# plt.legend(lns,labels,loc=7)
# plt.savefig("content_y_loss_compare.png", format='png')
# plt.show()


# plant content_loss
# loss = plaint_line(loss_contents)
# loss1,loss2,loss3,loss4 = pickle.load(open('loss.pkl'))
# losses = [loss1,loss2,loss3,loss4,loss]
# with open('loss.pkl', 'w')as f:
#     pickle.dump(losses, f)
# fig, ax1 = plt.subplots()
# lns1 = ax1.plot(np.arange(len(loss)),loss,label = "y_loss")
# ax1.set_xlabel("iter")
# ax1.set_ylabel("y_loss")
# plt.legend(lns1,loc=0)
# plt.savefig("y_loss.png", format='png')
# plt.show()
#
# ###############################plant time spend image##############################
# time56 = pickle.load(open('./images/56.pkl'))
# time112 = pickle.load(open('./images/112.pkl'))
# time128 = pickle.load(open('./images/128.pkl'))
# time200 = pickle.load(open('./images/200.pkl'))
# time224 = pickle.load(open('./images/224.pkl'))
# time250 = pickle.load(open('./images/250.pkl'))
# time300 = pickle.load(open('./images/300.pkl'))
# time350 = pickle.load(open('./images/350.pkl'))
# time400 = pickle.load(open('./images/400.pkl'))
# time4001 = pickle.load(open('./images/time5.pkl'))
# time448 = pickle.load(open('./images/448.pkl'))
# time512 = pickle.load(open('./images/512.pkl'))
# time600 = pickle.load(open('./images/600.pkl'))
# time650 = pickle.load(open('./images/650.pkl'))
# time896 = pickle.load(open('./images/896.pkl'))
#
# mean56 = np.mean(time56)
# mean112 = np.mean(time112)
# mean128 = np.mean(time128)
# mean200 = np.mean(time200)
# mean224 = np.mean(time224)
# mean250 = np.mean(time250)
# mean300 = np.mean(time300)
# mean350 = np.mean(time350)
# mean400 = np.mean(time400)
# mean4001 = np.mean(time4001)
# mean448 = np.mean(time448)
# mean512 = np.mean(time512)
# mean600 = np.mean(time600)
# mean650 = np.mean(time650)
# mean896 = np.mean(time896)
#
# mean = [mean56,mean112,mean128,mean200,mean224,mean250,mean300,mean350,mean400,mean448,mean512,mean600,mean650,mean4001]#,mean896
# x = [56,112,128,200,224,250,300,350,400,448,512,600,650,800]#,896
# plt.figure()
# plt.scatter(x,mean,marker="o",color='blue',linewidth=1,label="sssss")
# # plt.plot(x,mean,marker="o",color='blue',linewidth=1,label="sssss")
# plt.legend(loc=2)
# plt.xlabel("iteration")
# plt.ylabel("The value of time")
# plt.title("time")
# plt.savefig('mean_time.pdf',bbox_inches='tight',dpi=1024)
#
# plt.figure(figsize=(10,6))
# plt.plot(time56,color='blue',linewidth=1,label="56*56")
# plt.plot(time112,color='red',linewidth=1,label="112*112")
# plt.plot(time224,color='y',linewidth=1,label="224*224")
# plt.plot(time448,color='gray',linewidth=1,label="448*448")
# plt.plot(time896,color='black',linewidth=1,label="896*896")
# plt.legend(loc=1)
# plt.xlabel("iteration")
# plt.ylabel("The value of time")
# # plt.xticks(np.arange(0,50,1))
# plt.title("time")
# plt.savefig('time.pdf',bbox_inches='tight',dpi=1024)
# ###############################plant time spend image##############################
#

###############################30 images to different losses compose##############################
loss1,loss2,loss3,loss4,loss5,loss6= pickle.load(open('./Objective-evaluation-master/norm_ssim30before.pkl'))
loss1111,loss2222,loss3333,loss4444,loss5555,loss6666= pickle.load(open('./Objective-evaluation-master/norm_psnr30before.pkl'))
# loss11,loss22,loss33 = pickle.load(open('./Objective-evaluation-master/30_jiaozhi_ssim2.pkl'))
# loss44,loss55,loss66 = pickle.load(open('./Objective-evaluation-master/30_jiaozhi_psnr2.pkl'))
##jiaozhi###
# plt.subplot(311)
plt.figure(figsize=(10,6))
plt.plot(loss4,color='r',linewidth=1,label="Glioma(L_color+L_Y)")
plt.plot(loss5,color='y',linewidth=1,label="Glioma(L_color+L_Y+L_swap)")
plt.plot(loss6,color='b',linewidth=1,label="Glioma(L_color+L_Y+L_swap+L_c)")
plt.plot(loss1[0],color='g',linewidth=1,label="ADNI(L_color+L_Y)")
plt.plot(loss2,color='grey',linewidth=1,label="ADNI(L_color+L_Y+L_swap)")
plt.plot(loss3,color='black',linewidth=1,label="ADNI(L_color+L_Y+L_swap+L_c)")
# plt.plot(loss3,color='g',linewidth=1,label="(ult)L_color+L_Y")
# plt.plot(loss6,color='grey',linewidth=1,label="(ult)L_color+L_Y+L_swap")
# plt.plot(loss9,color='black',linewidth=1,label="(ult)L_color+L_Y+L_swap+L_c")
# ##ADNI###
# plt.subplot(312)
# plt.plot(loss2,color='r',linewidth=1,label="L_color+L_Y")
# plt.plot(loss5,color='g',linewidth=1,label="L_color+L_Y+L_styleswap")
# plt.plot(loss7,color='black',linewidth=1,label="L_color+L_Y+L_styleswap+L_c")

## plt.plot(loss6,color='black',linewidth=2,label="ADNI(L_c+L_color+L_y+L_cswap)")
# ##ult###
# plt.subplot(313)
# plt.plot(loss3,color='r',linewidth=1,label="L_color+L_Y")
# plt.plot(loss6,color='g',linewidth=1,label="L_color+L_Y+L_styleswap")
# plt.plot(loss9,color='y',linewidth=1,label="L_color+L_Y+L_styleswap+L_c")
plt.legend(loc=4)
plt.xlabel("index number")
plt.ylabel("SSIM")
plt.title("SSIMs of images")
plt.savefig('compose1.pdf',bbox_inches='tight',dpi=1024)

plt.figure(figsize=(10,6))
plt.plot(loss4444,color='r',linewidth=1,label="Glioma(L_color+L_Y)")
plt.plot(loss5555,color='y',linewidth=1,label="Glioma(L_color+L_Y+L_swap)")
plt.plot(loss6666,color='b',linewidth=1,label="Glioma(L_color+L_Y+L_swap+L_c)")
plt.plot(loss1111[0],color='g',linewidth=1,label="ADNI(L_color+L_Y)")
plt.plot(loss2222,color='grey',linewidth=1,label="ADNI(L_color+L_Y+L_swap)")
plt.plot(loss3333,color='black',linewidth=1,label="ADNI(L_color+L_Y+L_swap+L_c)")
plt.legend(loc=4)
plt.xlabel("index number")
plt.ylabel("PSNR")
plt.title("PSNRs of images")
plt.savefig('composepsnr.pdf',bbox_inches='tight',dpi=1024)
###############################30 images to different losses compose##############################



# ###############################plant loss Convergence image##############################
# def plaint_line(loss_contents):
#     l = len(loss_contents)
#     n = 45
#     batch_line = l/n
#     loss = []
#     for i in range(0,l,batch_line):
#         a = np.mean(loss_contents[i:i+batch_line])
#         loss.append(a)
#     return loss
#
# def de(i):
#     return i/1e3
# def de2(i):
#     return i/1e6
# loss1,loss2,loss3,loss4= pickle.load(open('loss.pkl'))
# # loss_1,loss_2,loss_3,loss_4,loss_5 = pickle.load(open('loss_onetime2.pkl'))
# print(np.shape(loss1))
# print(np.shape(loss2))
# print(np.shape(loss3))
# print(np.shape(loss4))
# loss3 = map(de,loss3)
# loss4 = map(de2,loss4)
# #
# # loss1_part = loss1[5:50]
# # loss2_part = loss2[5:50]
# # loss3_part = loss3[5:50]
# # loss4_part = loss4[5:50]
# # loss5_part = loss5[5:50]
# # loss_5_part = loss_5[5:50]
# #
# # loss1 = plaint_line(loss1_part)
# # loss2 = plaint_line(loss2_part)
# # loss3 = plaint_line(loss3_part)
# # loss4 = plaint_line(loss4_part)
# # loss5 = plaint_line(loss5_part)
# # loss_5 = plaint_line(loss_5_part)
#
# plt.figure(figsize=(10,6))
#
# plt.plot(loss1[0],color='r',linewidth=2,label="L_content(L_color+L_c)")
# plt.plot(loss2,color='g',linewidth=2,label="L_m(L_color+L_m)")
# plt.plot(loss3,color='gray',linewidth=2,label="Y-loss(MSE)/1e3(L_color+L_Y)")
# plt.plot(loss4,color='y',linewidth=2,label="Y-loss(L1)/1e6(L_color+L_Y)")
#
# plt.legend(loc=1)
# plt.xlabel("iteration")
# plt.ylabel("The value of losses")
# # plt.xticks(np.arange(0,50,1))
# plt.title("Convergence of losses")
# plt.savefig('Convergence of losses(part).pdf',bbox_inches='tight',dpi=1024)
#
# ###############################plant loss Convergence image###############################################

# # plant content_loss
# loss = plaint_line(loss_swap)
# loss1 = plaint_line(loss_yloss)
# losses = [loss, loss1]
# with open('swap_yloss_loss.pkl', 'w')as f:
#     pickle.dump(losses, f)
#
#
# plt.figure(figsize=(10, 6))
# plt.plot(loss,color='r',linewidth=2,label="loss_swap")
# plt.plot(loss1,color='g',linewidth=2,label="loss_yloss")
# plt.legend(loc=1)
# plt.xlabel("index number")
# plt.ylabel("loss value")
# plt.xticks(np.arange(0,15,1))
# plt.title("losses of images")
# plt.savefig('PSNRs_images.jpg',bbox_inches='tight',dpi=1024)