"""
render_3d.py — FPV Racing Drone Visualizer v4
Fixed: tunnel visible from frame 1, gates spawn ahead not on drone
"""
import sys, math, numpy as np, torch, pygame
from env_v2 import BioDroneEnv2
from model import FlyPolicyNetwork, get_device

SCREEN_W, SCREEN_H = 1000, 650
FPS_DEFAULT = 30
SAVE_PATH   = "weights/fly_policy.pth"
VP_X = SCREEN_W // 2
VP_Y = int(SCREEN_H * 0.38)
NEAR_HW = 340
NEAR_HH = 220
NEAR_Y  = int(SCREEN_H * 0.86)
NEAR_CX = SCREEN_W // 2
DRAW_DEPTH = 120
C_BG=(6,8,18); C_GRID=(16,20,34); C_FLOOR=(20,80,40); C_CEILING=(30,20,70)
C_WALL_L=(0,160,220); C_WALL_R=(220,50,70); C_GATE=(255,200,0)
C_GATE_PASS=(0,255,150); C_TEXT=(200,200,200); C_HL=(0,255,150)
C_SON=(0,255,150); C_SOFF=(35,38,55)
obs_global=[1.0]*7

def load_model(device):
    ck=torch.load(SAVE_PATH,map_location=device,weights_only=False)
    hp=ck['hyperparams']
    m=FlyPolicyNetwork(input_dim=hp['input_dim'],hidden_dim=hp['hidden_dim'],
                       output_dim=hp['output_dim'],sparsity=ck['sparsity']).to(device)
    m.load_state_dict(ck['model_state']); m.eval()
    print(f"Model loaded input={hp['input_dim']} best={ck['best_avg_reward']:.1f}")
    return m, ck

def fade(c, f):
    return tuple(max(0,min(255,int(ch*f//255))) for ch in c)

def project(wx, wy, t, tcx=0, tcy=0, thw=70, thh=80):
    d   = t**1.3
    vpx = VP_X - (tcx/max(thw,1))*26
    vpy = VP_Y + (tcy/max(thh,1))*14
    cx  = NEAR_CX + (vpx-NEAR_CX)*d
    cy  = NEAR_Y  + (vpy-NEAR_Y)*d
    hw  = NEAR_HW*(1-d)
    hh  = NEAR_HH*(1-d)
    sx  = cx + (wx/max(thw,1))*hw
    sy  = cy + (wy/max(thh,1))*hh
    return int(sx), int(sy)

def make_ring(t, cx, cy, hw, hh):
    return {
        'tl': project(-hw,-hh,t,cx,cy,hw,hh),
        'tr': project( hw,-hh,t,cx,cy,hw,hh),
        'bl': project(-hw, hh,t,cx,cy,hw,hh),
        'br': project( hw, hh,t,cx,cy,hw,hh),
        'c' : project(  0,  0,t,cx,cy,hw,hh)
    }

def draw_bg(screen):
    screen.fill(C_BG)
    for x in range(0,SCREEN_W,50):
        pygame.draw.line(screen,C_GRID,(x,0),(x,SCREEN_H),1)
    for y in range(0,SCREEN_H,50):
        pygame.draw.line(screen,C_GRID,(0,y),(SCREEN_W,y),1)

def draw_scene(screen, env, hist):
    # Pad history with current state if too short
    cur = (env.tunnel_center_x, env.tunnel_center_y,
           env.half_width, env.half_height)
    while len(hist) < DRAW_DEPTH+2:
        hist.insert(0, cur)

    n    = len(hist)
    segs = min(DRAW_DEPTH, n-1)

    def gh(t):
        i = max(0, min(n-1, int((1-t)*(n-1))))
        return hist[i]

    # Build rings near(0) to far(1)
    rings=[]
    for i in range(segs+1):
        t=i/segs
        cx,cy,hw,hh=gh(t)
        rings.append((t, make_ring(t,cx,cy,hw,hh), cx,cy,hw,hh))

    # Draw FAR to NEAR
    for i in range(len(rings)-2, -1, -1):
        t0,r0,cx0,cy0,hw0,hh0 = rings[i]
        t1,r1,cx1,cy1,hw1,hh1 = rings[i+1]
        nf  = 1.0-t0
        fv  = int(30+225*nf)
        lw  = max(1,int(3*nf))

        # Ceiling face
        pygame.draw.polygon(screen,(20,40,80),
            [r1['tl'],r1['tr'],r0['tr'],r0['tl']])
        # Floor face
        pygame.draw.polygon(screen,(15,60,25),
            [r1['bl'],r1['br'],r0['br'],r0['bl']])
        # Left wall face
        pygame.draw.polygon(screen,(20,20,70),
            [r1['tl'],r1['bl'],r0['bl'],r0['tl']])
        # Right wall face
        pygame.draw.polygon(screen,(20,20,70),
            [r1['tr'],r1['br'],r0['br'],r0['tr']])
        # Far cap fill
        pygame.draw.polygon(screen,(10,12,22),
            [r1['tl'],r1['tr'],r1['br'],r1['bl']])

        # Left wall lines (cyan)
        pygame.draw.line(screen,fade(C_WALL_L,fv),r1['tl'],r0['tl'],lw)
        pygame.draw.line(screen,fade(C_WALL_L,fv),r1['bl'],r0['bl'],lw)
        # Right wall lines (red)
        pygame.draw.line(screen,fade(C_WALL_R,fv),r1['tr'],r0['tr'],lw)
        pygame.draw.line(screen,fade(C_WALL_R,fv),r1['br'],r0['br'],lw)
        # Ceiling lines (purple)
        pygame.draw.line(screen,fade(C_CEILING,fv),r1['tl'],r0['tl'],lw)
        pygame.draw.line(screen,fade(C_CEILING,fv),r1['tr'],r0['tr'],lw)
        # Floor lines (green)
        pygame.draw.line(screen,fade(C_FLOOR,fv),r1['bl'],r0['bl'],lw)
        pygame.draw.line(screen,fade(C_FLOOR,fv),r1['br'],r0['br'],lw)

        # Ring cross lines (depth grid)
        if i % 5 == 0:
            pygame.draw.polygon(screen,fade((50,55,90),fv),
                [r0['tl'],r0['tr'],r0['br'],r0['bl']],1)

    # Back wall glow
    if rings:
        _,fr,*_ = rings[-1]
        pygame.draw.polygon(screen,(20,22,45),
            [fr['tl'],fr['tr'],fr['br'],fr['bl']])
        pygame.draw.polygon(screen,(40,45,90),
            [fr['tl'],fr['tr'],fr['br'],fr['bl']],1)
        pygame.draw.circle(screen,(80,110,220),fr['c'],6)
        pygame.draw.circle(screen,(160,200,255),fr['c'],3)

    # --- Draw Gates ---
    for gate in env.gates:
        if gate.dist_ahead <= 0 or gate.dist_ahead > DRAW_DEPTH*2:
            continue
        t   = min(gate.dist_ahead / DRAW_DEPTH, 0.95)
        cx,cy,hw,hh = gh(t)
        cl  = 1.0 - t
        fv  = int(40+215*cl)
        lw  = max(1, int(5*cl))

        if gate.passed:
            gc = fade(C_GATE_PASS, fv)
        elif gate.dist_ahead <= 6:
            fl = int(abs(math.sin(pygame.time.get_ticks()*0.012))*255)
            gc = (255, fl, 0)
        else:
            gc = fade(C_GATE, fv)

        gx  = gate.center_x
        gy  = gate.center_y
        gw  = gate.gap_width  / 2
        gh_ = gate.gap_height / 2
        fm  = max(6.0, gw*0.2)

        otl=project(gx-gw-fm, gy-gh_-fm, t, cx,cy,hw,hh)
        otr=project(gx+gw+fm, gy-gh_-fm, t, cx,cy,hw,hh)
        obl=project(gx-gw-fm, gy+gh_+fm, t, cx,cy,hw,hh)
        obr=project(gx+gw+fm, gy+gh_+fm, t, cx,cy,hw,hh)
        itl=project(gx-gw,    gy-gh_,    t, cx,cy,hw,hh)
        itr=project(gx+gw,    gy-gh_,    t, cx,cy,hw,hh)
        ibl=project(gx-gw,    gy+gh_,    t, cx,cy,hw,hh)
        ibr=project(gx+gw,    gy+gh_,    t, cx,cy,hw,hh)

        # Gate frame fill (dark gold)
        pygame.draw.polygon(screen,(40,32,4),[otl,otr,itr,itl])
        pygame.draw.polygon(screen,(40,32,4),[ibl,ibr,obr,obl])
        pygame.draw.polygon(screen,(40,32,4),[otl,itl,ibl,obl])
        pygame.draw.polygon(screen,(40,32,4),[itr,otr,obr,ibr])
        # Gate border
        pygame.draw.polygon(screen,gc,[otl,otr,obr,obl],lw)
        pygame.draw.polygon(screen,gc,[itl,itr,ibr,ibl],lw)
        # Corner dots
        for pt in [otl,otr,obl,obr]:
            pygame.draw.circle(screen,gc,pt,max(2,int(5*cl)))

        # Distance label when close
        if gate.dist_ahead < 25 and cl > 0.3:
            ft_=pygame.font.SysFont("monospace",11)
            lb=ft_.render(f"GATE {gate.dist_ahead}",True,gc)
            screen.blit(lb,((otl[0]+otr[0])//2-lb.get_width()//2,
                            min(otl[1],otr[1])-14))

    # --- Near plane mask (darken outside tunnel) ---
    if rings:
        _,nr,*_ = rings[0]
        tl,tr,bl,br = nr['tl'],nr['tr'],nr['bl'],nr['br']
        # Outside strips
        pygame.draw.rect(screen,C_BG,(0,0,tl[0],SCREEN_H))
        pygame.draw.rect(screen,C_BG,(tr[0],0,SCREEN_W-tr[0],SCREEN_H))
        pygame.draw.rect(screen,C_BG,(tl[0],0,tr[0]-tl[0],min(tl[1],tr[1])))
        pygame.draw.rect(screen,C_BG,(bl[0],max(0,bl[1]),
                                      br[0]-bl[0],SCREEN_H-bl[1]))
        # Near ring border
        pygame.draw.polygon(screen,(80,90,150),[tl,tr,br,bl],2)

def draw_sensors(screen, obs):
    cx,cy = NEAR_CX, int(NEAR_Y*0.68)
    # Horizontal rays
    for i,angle in enumerate(BioDroneEnv2.SENSOR_ANGLES_H):
        rl = float(obs[i])*200
        sa = math.radians(-angle-90)
        ex = cx+rl*math.cos(sa)
        ey = cy+rl*math.sin(sa)
        d  = 1.0-float(obs[i])
        col= (255,int(220*(1-d)),int(50*(1-d**2)))
        pygame.draw.line(screen,col,(cx,cy),(int(ex),int(ey)),1)
        pygame.draw.circle(screen,col,(int(ex),int(ey)),3)
    # Up sensor
    ul=float(obs[5])*90
    cu=(255,int(220*(1-float(obs[5]))),50)
    pygame.draw.line(screen,cu,(cx,cy),(cx,int(cy-ul)),1)
    pygame.draw.circle(screen,cu,(cx,int(cy-ul)),3)
    # Down sensor
    dl=float(obs[6])*90
    cd=(255,int(220*(1-float(obs[6]))),50)
    pygame.draw.line(screen,cd,(cx,cy),(cx,int(cy+dl)),1)
    pygame.draw.circle(screen,cd,(cx,int(cy+dl)),3)

def draw_reticle(screen, action):
    cx,cy = NEAR_CX, int(NEAR_Y*0.68)
    cols={0:(0,200,255),1:(0,255,150),2:(255,160,50),
          3:(200,100,255),4:(255,80,120)}
    col=cols.get(action,(0,255,150))
    # Outer rings
    pygame.draw.circle(screen,col,(cx,cy),28,1)
    pygame.draw.circle(screen,fade(col,120),(cx,cy),36,1)
    # Centre dot
    pygame.draw.circle(screen,col,(cx,cy),5)
    # Cross
    g,a=11,24
    pygame.draw.line(screen,col,(cx-a,cy),(cx-g,cy),2)
    pygame.draw.line(screen,col,(cx+g,cy),(cx+a,cy),2)
    pygame.draw.line(screen,col,(cx,cy-a),(cx,cy-g),2)
    pygame.draw.line(screen,col,(cx,cy+g),(cx,cy+a),2)
    # Forward line to VP
    pygame.draw.line(screen,fade(col,60),(cx,cy-32),(VP_X,VP_Y+8),1)

def draw_connectome(screen, model, fs, ft):
    px,py,pw,ph=15,SCREEN_H-158,210,148
    s=pygame.Surface((pw,ph),pygame.SRCALPHA)
    s.fill((12,15,30,220)); screen.blit(s,(px,py))
    pygame.draw.rect(screen,(50,55,85),(px,py,pw,ph),1)
    screen.blit(fs.render("Connectome",True,C_HL),(px+8,py+6))
    m1=model.fc1.mask.cpu().numpy()
    m2=model.fc2.mask.cpu().numpy()
    for r in range(m1.shape[0]):
        for c in range(m1.shape[1]):
            col=C_SON if m1[r,c]>0 else C_SOFF
            pygame.draw.circle(screen,col,(px+20+c*9,py+32+r*3),2)
    for r in range(m2.shape[0]):
        for c in range(m2.shape[1]):
            col=C_SON if m2[r,c]>0 else C_SOFF
            pygame.draw.circle(screen,col,(px+20+c*4,py+64+r*12),2)
    a1=int(model.fc1.mask.sum())
    a2=int(model.fc2.mask.sum())
    screen.blit(ft.render(f"Active:{a1}/160 {a2}/96",True,C_TEXT),
                (px+8,py+132))

def draw_sensor_bars(screen, obs, fs, ft):
    labels=["FL","NL","FW","NR","FR","UP","DN"]
    sx,sy=SCREEN_W-200,SCREEN_H-185
    s=pygame.Surface((185,178),pygame.SRCALPHA)
    s.fill((12,15,30,220)); screen.blit(s,(sx-5,sy-5))
    pygame.draw.rect(screen,(50,55,85),(sx-5,sy-5,185,178),1)
    screen.blit(fs.render("Sensor Readings",True,C_HL),(sx,sy-2))
    for i,(lb,v) in enumerate(zip(labels,obs)):
        by=sy+18+i*22; bw=int(float(v)*130)
        d=1.0-float(v); col=(255,int(220*(1-d)),50)
        screen.blit(ft.render(lb,True,C_TEXT),(sx,by+3))
        pygame.draw.rect(screen,(35,38,58),(sx+22,by,130,14))
        if bw>0: pygame.draw.rect(screen,col,(sx+22,by,bw,14))
        screen.blit(ft.render(f"{v:.2f}",True,C_TEXT),(sx+156,by+2))

def draw_hud(screen,fl,fs,ft,ep,rw,st,act,paused,best,fps,gp):
    anames={0:"◀ LEFT",1:"▲ FWD",2:"▶ RIGHT",3:"↑ UP",4:"↓ DOWN"}
    acols={0:(0,200,255),1:(0,255,150),2:(255,160,50),
           3:(200,100,255),4:(255,80,120)}
    b=pygame.Surface((SCREEN_W,46),pygame.SRCALPHA)
    b.fill((6,8,20,240)); screen.blit(b,(0,0))
    pygame.draw.line(screen,(38,44,72),(0,46),(SCREEN_W,46),1)
    screen.blit(fl.render(f"Episode {ep}",True,C_HL),(12,6))
    screen.blit(fl.render(f"Reward:{rw:.0f}",True,C_TEXT),(220,6))
    screen.blit(fs.render(f"Step:{st}",True,C_TEXT),(450,8))
    screen.blit(fs.render(f"Gates:{gp}",True,(255,200,0)),(550,8))
    bs=f"Best:{best:.0f}" if best>-1e9 else "Best:--"
    screen.blit(fs.render(bs,True,C_HL),(450,28))
    screen.blit(ft.render(f"FPS:{fps:.0f}",True,(70,80,110)),(550,28))
    an=fl.render(anames.get(act,"?"),True,acols.get(act,C_TEXT))
    screen.blit(an,(SCREEN_W-an.get_width()-14,8))
    ms=float(min(obs_global)); dng=max(0.0,1.0-ms); bw=int(SCREEN_W*dng)
    if bw>0:
        dc=(int(255*dng),int(80*(1-dng)),30)
        ds=pygame.Surface((bw,3),pygame.SRCALPHA)
        ds.fill((*dc,210)); screen.blit(ds,(0,43))
    if paused:
        ov=pygame.Surface((SCREEN_W,SCREEN_H),pygame.SRCALPHA)
        ov.fill((0,0,0,150)); screen.blit(ov,(0,0))
        pt=fl.render("PAUSED — SPACE to resume",True,(255,255,80))
        screen.blit(pt,(SCREEN_W//2-pt.get_width()//2,SCREEN_H//2-16))
    ht=ft.render("Q:Quit  SPACE:Pause  R:Restart  +/-:Speed",
                 True,(50,58,80))
    screen.blit(ht,(SCREEN_W//2-ht.get_width()//2,SCREEN_H-16))

def run():
    global obs_global
    print("="*55)
    print("  BioDrone-RL  FPV Gate Navigation  v4")
    print("="*55)
    device=get_device()
    model,ck=load_model(device)
    env=BioDroneEnv2()
    mid=ck['hyperparams']['input_dim']
    print(f"Model input:{mid}  Env sensors:7")
    if mid==5: print("Trimming obs to 5 (old model — retrain for full 7)")

    pygame.init()
    screen=pygame.display.set_mode((SCREEN_W,SCREEN_H))
    pygame.display.set_caption("BioDrone-RL | FPV Gate Navigation v4")
    clock=pygame.time.Clock()
    fl=pygame.font.SysFont("monospace",16,bold=True)
    fs=pygame.font.SysFont("monospace",12)
    ft=pygame.font.SysFont("monospace",11)

    def new_ep():
        o,_=env.reset()
        # Pre-fill history so tunnel is visible immediately
        cur=(env.tunnel_center_x,env.tunnel_center_y,
             env.half_width,env.half_height)
        h=[cur]*(DRAW_DEPTH+2)
        return o,0.0,0,False,h,0

    obs,rw,st,done,hist,gp=new_ep()
    obs_global=list(obs)
    ep=1; best=-float('inf'); paused=False
    fps_t=FPS_DEFAULT; fps_a=float(FPS_DEFAULT); action=1

    while True:
        for ev in pygame.event.get():
            if ev.type==pygame.QUIT:
                env.close(); pygame.quit(); sys.exit()
            elif ev.type==pygame.KEYDOWN:
                if ev.key in(pygame.K_q,pygame.K_ESCAPE):
                    env.close(); pygame.quit(); sys.exit()
                elif ev.key==pygame.K_SPACE:
                    paused=not paused
                elif ev.key==pygame.K_r:
                    obs,rw,st,done,hist,gp=new_ep()
                    obs_global=list(obs)
                elif ev.key in(pygame.K_PLUS,pygame.K_EQUALS,
                               pygame.K_KP_PLUS):
                    fps_t=min(120,fps_t+10)
                elif ev.key in(pygame.K_MINUS,pygame.K_KP_MINUS):
                    fps_t=max(5,fps_t-10)

        if not paused:
            if done:
                pygame.time.wait(500)
                if rw>best: best=rw
                ep+=1
                obs,rw,st,done,hist,gp=new_ep()
                obs_global=list(obs)
            else:
                oi=obs[:mid]
                with torch.no_grad():
                    ot=torch.tensor(oi,dtype=torch.float32).to(device)
                    action,_=model.get_action(ot)
                # Append current wall state BEFORE stepping
                hist.append((env.tunnel_center_x,env.tunnel_center_y,
                             env.half_width,env.half_height))
                if len(hist)>DRAW_DEPTH+10: hist.pop(0)
                obs,reward,term,trunc,info=env.step(action)
                obs_global=list(obs)
                rw+=reward; st+=1
                gp=info.get("gates_passed",gp)
                done=term or trunc

        draw_bg(screen)
        draw_scene(screen,env,hist)
        draw_sensors(screen,obs)
        draw_reticle(screen,action)
        draw_connectome(screen,model,fs,ft)
        draw_sensor_bars(screen,obs,fs,ft)
        draw_hud(screen,fl,fs,ft,ep,rw,st,action,
                 paused,best,fps_a,gp)
        pygame.display.flip()
        dt=clock.tick(fps_t)
        fps_a=1000.0/max(dt,1)

if __name__=="__main__":
    run()
