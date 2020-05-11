from plotfig import plot2D,plot1D
import numpy as np

pix_wid = 0.675

#def edge_lineplot(images,save_filen,legend,linestyle,markstyle,filen,x_idx,y_idx):
#    if 'horz' in filen:
#        return horz_edge_lineplot(images,save_filen,legend,linestyle,markstyle,x_idx,y_idx)
#    if 'vert' in filen:
#        return vert_edge_lineplot(images,save_filen,legend,linestyle,markstyle,x_idx,y_idx)

def horz_edge_lineplot(images,save_filen,legend,linestyle,markstyle,markevery,x_idx,y_idx,mark2D=False,aspect=0.8):
    sh = images[0].shape
    #x = np.arange(-images[0].shape[1]//2,images[0].shape[1]//2,1)*pix_wid
    #y = np.arange(images[0].shape[0]//2,-images[0].shape[0]//2,-1)*pix_wid
    x = np.arange(0,images[0].shape[1],1)*pix_wid
    y = np.arange(images[0].shape[0],0,-1)*pix_wid
    draw_rect = [[(x[0],y[-1]),((sh[1]//4)*pix_wid,(sh[0]//4)*pix_wid)],
                 [(x[-1]-(sh[1]//4)*pix_wid,y[sh[0]//4]),((sh[1]//4)*pix_wid,(sh[0]//4)*pix_wid)],
                 [(x[0],y[sh[1]//4]),((sh[1]//4)*pix_wid,(sh[0]//4)*pix_wid)],
                 [(x[-1]-(sh[1]//4)*pix_wid,y[-1]),((sh[1]//4)*pix_wid,(sh[0]//4)*pix_wid)]]
    lines = []
    for j,(xid,yid,leg) in enumerate(zip(x_idx,y_idx,legend)):
        lines.append([np.ones(len(yid))*x[xid],y[yid]])
        x_list,y_list = [],[]
        for i,img in enumerate(images):
            x_list.append(y[yid])
            y_list.append(img[yid,xid])
        plot1D(x_list,y_list,save_filen+'_line{}.png'.format(j),aspect=aspect,legend=leg,linesty=linestyle,markstyle=markstyle[j],markevery=markevery[j])

    if mark2D:
        plot2D(x,y,images[0],save_filen+'_img.png',lines=lines,linesty=['r-']*len(lines),draw_rect=draw_rect)  
    else: 
        plot2D(x,y,images[0],save_filen+'_img.png',lines=lines,linesty=['r-']*len(lines))  
    
def vert_edge_lineplot(images,save_filen,legend,linestyle,markstyle,markevery,x_idx,y_idx,mark2D=False,aspect=0.8):
    sh = images[0].shape
    #x = np.arange(-images[0].shape[1]//2,images[0].shape[1]//2,1)*pix_wid
    #y = np.arange(images[0].shape[0]//2,-images[0].shape[0]//2,-1)*pix_wid
    x = np.arange(0,images[0].shape[1],1)*pix_wid
    y = np.arange(images[0].shape[0],0,-1)*pix_wid
    draw_rect = [[(x[0],y[-1]),((sh[1]//4)*pix_wid,(sh[0]//4)*pix_wid)],
                 [(x[-1]-(sh[1]//4)*pix_wid,y[sh[0]//4]),((sh[1]//4)*pix_wid,(sh[0]//4)*pix_wid)],
                 [(x[0],y[sh[1]//4]),((sh[1]//4)*pix_wid,(sh[0]//4)*pix_wid)],
                 [(x[-1]-(sh[1]//4)*pix_wid,y[-1]),((sh[1]//4)*pix_wid,(sh[0]//4)*pix_wid)]]
    lines = []
    for j,(xid,yid,leg) in enumerate(zip(x_idx,y_idx,legend)):
        lines.append([x[xid],np.ones(len(xid))*y[yid]])
        x_list,y_list = [],[]
        for i,img in enumerate(images):
            x_list.append(x[xid])
            y_list.append(img[yid,xid])
        plot1D(x_list,y_list,save_filen+'_line{}.png'.format(j),aspect=aspect,legend=leg,linesty=linestyle,markstyle=markstyle[j],markevery=markevery[j])

    if mark2D:
        plot2D(x,y,images[0],save_filen+'_img.png',lines=lines,linesty=['r-']*len(lines),draw_rect=draw_rect)  
    else: 
        plot2D(x,y,images[0],save_filen+'_img.png',lines=lines,linesty=['r-']*len(lines))  
    
