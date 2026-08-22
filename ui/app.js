const evidence = window.WING_SPECTRUM_EVIDENCE;
document.getElementById('version').textContent = evidence.version;
document.getElementById('run-link').href = evidence.runUrl;
const chart = document.getElementById('chart-area');
const buttons = [...document.querySelectorAll('[data-view]')];
const fmt = n => Number(n).toFixed(4);
const final = run => run.history[run.history.length - 1];

function metricsView(){
  const f=final(evidence.faulty), x=final(evidence.fixed);
  const rows=[['UVS visibility','uvs_visibility',Math.max(f.uvs_visibility,x.uvs_visibility)*1.12,'delta'],['VS visibility','vs_visibility',Math.max(f.vs_visibility,x.vs_visibility)*1.12,'delta'],['Human reflectance','human_reflectance',.2,'within bound'],['Solar transmittance','solar_transmittance',1,'trade-off']];
  chart.innerHTML=`<div class="legend"><span><i class="faulty"></i>Faulty VJP</span><span><i class="fixed"></i>Fixed VJP</span></div><div class="metric-chart">${rows.map(([label,key,max,note])=>{const delta=(x[key]-f[key])/f[key]*100;const result=note==='delta'?`${delta>=0?'+':''}${delta.toFixed(1)}%`:note;return `<div class="metric-row"><div class="metric-label">${label}</div><div class="bar-pair"><div class="bar-line"><div class="bar-fill faulty" style="width:${f[key]/max*100}%">${fmt(f[key])}</div></div><div class="bar-line"><div class="bar-fill fixed" style="width:${x[key]/max*100}%">${fmt(x[key])}</div></div></div><div class="metric-delta ${note==='delta'?'':'neutral'}">${result}</div></div>`}).join('')}</div>`;
}
function linePath(values,w,h,pad,min,max){return values.map((v,i)=>`${i?'L':'M'}${pad+i/(values.length-1)*(w-2*pad)},${h-pad-(v-min)/(max-min)*(h-2*pad)}`).join(' ')}
function plotView(kind){
  const W=1100,H=360,P=52; let xs,series,title,ylabel;
  if(kind==='spectrum'){xs=evidence.wavelengths;series=[['Initial',evidence.fixed.initialReflectance,'initial-line'],['Faulty',evidence.faulty.reflectance,'faulty-line'],['Fixed',evidence.fixed.reflectance,'fixed-line']];title='Final reflectance spectrum';ylabel='Reflectance'}
  else{xs=evidence.fixed.history.map(d=>d.step);series=[['Faulty',evidence.faulty.history.map(d=>d.loss),'faulty-line'],['Fixed',evidence.fixed.history.map(d=>d.loss),'fixed-line']];title='Optimization loss';ylabel='Loss'}
  const vals=series.flatMap(s=>s[1]), min=Math.min(...vals),max=Math.max(...vals), ticks=[0,.25,.5,.75,1];
  chart.innerHTML=`<div class="legend">${series.map(s=>`<span><i class="${s[2].replace('-line','')}"></i>${s[0]}</span>`).join('')}</div><svg class="plot" viewBox="0 0 ${W} ${H}" role="img" aria-label="${title}"><text x="${P}" y="18">${title}</text>${ticks.map(t=>`<line class="grid" x1="${P}" y1="${H-P-t*(H-2*P)}" x2="${W-P}" y2="${H-P-t*(H-2*P)}"/><text x="6" y="${H-P-t*(H-2*P)+4}">${(min+t*(max-min)).toFixed(2)}</text>`).join('')}<line class="axis" x1="${P}" y1="${H-P}" x2="${W-P}" y2="${H-P}"/>${series.map(s=>`<path class="${s[2]}" d="${linePath(s[1],W,H,P,min,max)}"/>`).join('')}<text x="${W/2}" y="${H-12}" text-anchor="middle">${kind==='spectrum'?'Wavelength (nm)':'Iteration'}</text><text transform="translate(14 ${H/2}) rotate(-90)" text-anchor="middle">${ylabel}</text><text x="${P}" y="${H-32}">${xs[0]}</text><text x="${W-P}" y="${H-32}" text-anchor="end">${xs[xs.length-1]}</text></svg>`;
}
buttons.forEach(button=>button.addEventListener('click',()=>{buttons.forEach(b=>b.classList.toggle('active',b===button));button.dataset.view==='metrics'?metricsView():plotView(button.dataset.view)}));
metricsView();
