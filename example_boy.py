class Fencer:
    def __init__(self,lunge=False,duck=False,toe_touch=False,flech=False):
        self.lunge=lunge
        self.duck=duck
        self.toe_touch=toe_touch
        self.flech=flech
    def same_frame(self,list):
        
        #create pose dict here
        #after creating, append that pose dict to the list parameter
        list.append([("lunge="+str(self.lunge)),("duck="+str(self.duck)),("toe_touch="+str(self.toe_touch)),("flech="+str(self.flech))])        


F1_ze_movie=Fencer()
feedback=[]
for i in range(0,160):
    
    F1_ze_movie.same_frame(feedback)
# print(feedback)
print()
numerals=(int(input("choose a number 1 through 161: ")))
# print()
# input=(int(input("Do you wish to check for 1 for lunge, 2 for duck, 3 for toe touch, or 4 for flech: ")))
print()
for key, value in feedback[int(numerals-1).items]:
    print(key,value)
print()

# input-=1
# numerals-=1
# print(feedback[numerals][input])
print()
'''
using these index positions to find data sucks

what if you have new pose flags?

what if you dont know the index positions?

old style:
print(feedback[49][2])

new style (todays goal):
print(feedback[49][toe_touch])



'''