// Плеер: один аудиофайл на ленту, проигрывание отрезка через seek + остановку
// на конце. Так не нужно резать сотни клипов на диск, а переход между
// сегментами мгновенный.
const TA = {
  audio: null, stopAt: null, limit: true,
  fmt(t){ t=Math.max(0,t); const m=Math.floor(t/60), s=t-m*60;
    return `${String(m).padStart(2,'0')}:${s.toFixed(1).padStart(4,'0')}`; },
  init(){
    this.audio = document.getElementById('player');
    if(!this.audio) return;
    this.audio.addEventListener('timeupdate', ()=>{
      const t=this.audio.currentTime;
      document.getElementById('now').textContent =
        `${this.fmt(t)}` + (this.stopAt!==null?` → ${this.fmt(this.stopAt)}`:'');
      if(this.limit && this.stopAt!==null && t>=this.stopAt){ this.audio.pause(); this.stopAt=null; }
      this.drawCursor(t);
    });
    const lim=document.getElementById('limit');
    if(lim){ lim.addEventListener('click',()=>{ this.limit=!this.limit;
      lim.classList.toggle('on',this.limit);
      lim.textContent = this.limit?'останавливать на конце сегмента':'играть дальше'; }); }
    document.addEventListener('keydown',e=>{
      if(e.target.tagName==='INPUT'||e.target.tagName==='TEXTAREA') return;
      if(e.code==='Space'){ e.preventDefault(); this.audio.paused?this.audio.play():this.audio.pause(); }
      if(e.key==='ArrowLeft') this.audio.currentTime-=5;
      if(e.key==='ArrowRight') this.audio.currentTime+=5;
    });
    this.loadMarks(); this.wire(); this.drawAll(); this.jumpToHash();
    window.addEventListener('resize', ()=>this.drawAll());
  },
  jumpToHash(){
    // переход со страницы песен: подсветить нужный сегмент и встать на его начало
    if(!location.hash) return;
    const tr=document.getElementById(location.hash.slice(1));
    if(!tr) return;
    tr.scrollIntoView({block:'center'});
    tr.style.outline='2px solid var(--accent)';
    const btn=tr.querySelector('[data-play]');
    if(btn) this.audio.currentTime=parseFloat(btn.dataset.start);
  },
  play(a,b){ this.stopAt = (b!==undefined && b!==null) ? b : null;
    this.audio.currentTime=a; this.audio.play(); },
  wire(){
    document.querySelectorAll('[data-play]').forEach(el=>{
      el.addEventListener('click',()=>{
        const a=parseFloat(el.dataset.start), b=el.dataset.end?parseFloat(el.dataset.end):null;
        this.play(a,b);
      });
    });
    document.querySelectorAll('[data-mark]').forEach(el=>{
      el.addEventListener('click',()=>{
        const key=el.dataset.key, kind=el.dataset.mark;
        const m=this.marks[key]||{};
        m[kind]=!m[kind]; this.marks[key]=m; this.saveMarks();
        el.classList.toggle('on', !!m[kind]);
      });
    });
    const ex=document.getElementById('export');
    if(ex) ex.addEventListener('click',()=>{
      const blob=new Blob([JSON.stringify({marks:this.marks,exported:new Date().toISOString()},null,2)],
        {type:'application/json'});
      const a=document.createElement('a'); a.href=URL.createObjectURL(blob);
      a.download='tapearch-marks.json'; a.click();
    });
    const im=document.getElementById('importfile');
    if(im) im.addEventListener('change',e=>{
      const f=e.target.files[0]; if(!f) return;
      const r=new FileReader();
      r.onload=()=>{ try{ const d=JSON.parse(r.result);
        this.marks=Object.assign(this.marks,d.marks||d); this.saveMarks(); location.reload();
      }catch(err){ alert('не разобрал файл: '+err); } };
      r.readAsText(f);
    });
    document.querySelectorAll('[data-filter]').forEach(el=>{
      el.addEventListener('click',()=>{ el.classList.toggle('on'); this.applyFilters(); });
    });
    const map=document.getElementById('map');
    if(map) map.addEventListener('click',e=>{
      const r=map.getBoundingClientRect();
      this.play((e.clientX-r.left)/r.width*TA.duration, null);
    });
  },
  loadMarks(){
    try{ this.marks=JSON.parse(localStorage.getItem('tapearch.marks')||'{}'); }
    catch(e){ this.marks={}; }
    document.querySelectorAll('[data-mark]').forEach(el=>{
      const m=this.marks[el.dataset.key]||{};
      el.classList.toggle('on', !!m[el.dataset.mark]);
    });
  },
  saveMarks(){ localStorage.setItem('tapearch.marks', JSON.stringify(this.marks)); },
  applyFilters(){
    const grades=[...document.querySelectorAll('[data-filter="grade"].on')].map(e=>e.dataset.value);
    const cats=[...document.querySelectorAll('[data-filter="cat"].on')].map(e=>e.dataset.value);
    const starred=document.querySelector('[data-filter="star"].on');
    document.querySelectorAll('tr.seg').forEach(tr=>{
      let ok=true;
      if(grades.length && !grades.includes(tr.dataset.grade)) ok=false;
      if(cats.length && !cats.includes(tr.dataset.cat)) ok=false;
      if(starred){ const m=this.marks[tr.dataset.key]||{}; if(!m.star) ok=false; }
      tr.style.display = ok?'':'none';
    });
  },
  drawAll(){
    if(!window.TA_PEAKS) return;
    document.querySelectorAll('canvas.wf').forEach(c=>this.drawWave(c,
      parseFloat(c.dataset.start), parseFloat(c.dataset.end), c.dataset.color));
    const map=document.getElementById('map'); if(map) this.drawMap(map);
  },
  peakRange(a,b,w){
    const p=window.TA_PEAKS, fps=p.per_second, n=p.min.length;
    const i0=Math.max(0,Math.floor(a*fps)), i1=Math.min(n,Math.ceil(b*fps));
    const out=[];
    for(let x=0;x<w;x++){
      const s=i0+Math.floor((i1-i0)*x/w), e=Math.max(s+1,i0+Math.floor((i1-i0)*(x+1)/w));
      let lo=0,hi=0;
      for(let i=s;i<e && i<n;i++){ if(p.min[i]<lo)lo=p.min[i]; if(p.max[i]>hi)hi=p.max[i]; }
      out.push([lo/127,hi/127]);
    }
    return out;
  },
  drawWave(c,a,b,color){
    const dpr=window.devicePixelRatio||1, w=c.clientWidth, h=c.clientHeight;
    c.width=w*dpr; c.height=h*dpr;
    const g=c.getContext('2d'); g.scale(dpr,dpr); g.clearRect(0,0,w,h);
    const data=this.peakRange(a,b,w); g.fillStyle=color||'#d9a84e';
    for(let x=0;x<w;x++){
      const [lo,hi]=data[x];
      const y0=h/2-hi*h/2, y1=h/2-lo*h/2;
      g.fillRect(x,y0,1,Math.max(1,y1-y0));
    }
  },
  drawMap(c){
    const dpr=window.devicePixelRatio||1, w=c.clientWidth, h=c.clientHeight;
    c.width=w*dpr; c.height=h*dpr;
    const g=c.getContext('2d'); g.scale(dpr,dpr); g.clearRect(0,0,w,h);
    (window.TA_BANDS||[]).forEach(bd=>{
      g.fillStyle=bd.color+'44';
      g.fillRect(bd.start/TA.duration*w,0,Math.max(1,(bd.end-bd.start)/TA.duration*w),h);
    });
    const data=this.peakRange(0,TA.duration,w);
    for(let x=0;x<w;x++){
      const t=x/w*TA.duration;
      const bd=(window.TA_BANDS||[]).find(b=>t>=b.start&&t<b.end);
      g.fillStyle=bd?bd.color:'#8a8375';
      const [lo,hi]=data[x]; const y0=h/2-hi*h/2, y1=h/2-lo*h/2;
      g.fillRect(x,y0,1,Math.max(1,y1-y0));
    }
    this._map=c;
  },
  drawCursor(t){
    const c=this._map; if(!c) return;
    if(!this._mapImg){ this._mapImg=c.getContext('2d').getImageData(0,0,c.width,c.height); }
    const g=c.getContext('2d'); g.putImageData(this._mapImg,0,0);
    const dpr=window.devicePixelRatio||1;
    g.save(); g.scale(dpr,dpr);
    g.fillStyle='#fff'; g.fillRect(t/TA.duration*c.clientWidth,0,1,c.clientHeight);
    g.restore();
  },
};
document.addEventListener('DOMContentLoaded',()=>TA.init());
