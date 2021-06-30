import os
file_epl = open('expansion_loop','w')
file_cl = open('cnid_loop','w')
file_dl = open('deleting_loop','w')

size_list = [[1, 2], [1, 3], [1, 5], [1, 7], [1, 8],\
             [2, 2], [2, 3], [2, 5], [2, 7], [2, 8],\
             [3, 2], [3, 3], [3, 5], [3, 7], [3, 8],\
             [4, 7], [5, 5], [7, 7]]
for i in size_list:
    expansion_name = '{0}{1}'.format(i[0],i[1])
    os.chdir(expansion_name)
    for j in range(10):
        for k in range(58):
            cnid_name = '{0} {1}'.format(j,k)
            cnid_name_t = '{0}_{1}'.format(j,k)
            os.rename(cnid_name, cnid_name_t)
            os.chdir(cnid_name_t)
            for h in os.listdir():
                file_epl.write('{0}{1}\n'.format(i[0],i[1]))
                file_dl.write('{}\n'.format(h)) 
                file_cl.write('{} \n'.format(cnid_name_t))
            os.chdir(os.path.pardir)
    os.chdir(os.path.pardir)
file_epl.close()
file_cl.close()
file_dl.close()