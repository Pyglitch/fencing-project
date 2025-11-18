
frames_list = [
    [1, -5],
    [2, 10],
    [3, 2],
    [4, 9],
    [5, 5],
    [6, 5],
    [7, 5],
    [8, 5],
]

lean_list = [
    [1, "lean"],
    [2, "lean"],
    [3, "no lean"],
    [4, "no lean"],
    [5, "no lean"],
    [6, "lean"],
    [7, "lean"],
    [8, "lean"],
]
mex=0
mox_fram=1
#we can share index positions between the two lists
for i in range(len(lean_list)):
    # print(frames_list[i])
    # print(lean_list[i])
    if lean_list[i][1]=="no lean":
        if int(frames_list[i][1]) > mex:
            mex=int(frames_list[i][1])
            mox_fram=lean_list[i][0]
print(mex)
print(mox_fram)

#how do we compare the max?
max = frames_list[0][0]
#setting max to the FIRST value (of all the second inside values)
for line in frames_list:
    if line[1] > max:
        max = line[1]

#after the iteration, show the max
# print("max vx:",max)

'''
CHALLENGE 6/26
find the max only for the frame numbers that have "no lean" in the list entry

here is a small example of only printing the list item if it contains the flag "no lean"
'''
# for line in lean_list:
    # if "no lean" in line:
        # print(line)
        

