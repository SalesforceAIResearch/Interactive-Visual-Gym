/* Only reviewed feed data. textContent prevents model names becoming HTML. */
"use strict";
let entries = [], engine = "cooksim";
const status = document.querySelector("#status"), rows = document.querySelector("#rows");
const protocol = document.querySelector("#protocol"), details = document.querySelector("#details");
const pct = v => typeof v === "number" && Number.isFinite(v) ? `${(100 * v).toFixed(1)}%` : "—";
const families = {"frontier-api":"Frontier API", "streaming-api":"Streaming API", "open-nonstreaming":"Open-weight", "open-streaming":"Open-weight streaming"};
function node(tag, text, cls) { const n = document.createElement(tag); if(text !== undefined) n.textContent = text; if(cls) n.className=cls; return n; }
function detail(entry) {
  details.replaceChildren(node("h3", entry.assistant.name));
  const table=node("table"), head=node("tr");
  ["Run", "Episodes", "Errors scheduled → triggered", "Rounds reached → scored", "Questions reached → scored", "Requests reached → scored"].forEach(t=>head.append(node("th",t)));
  table.append(head);
  for(const [run,r] of Object.entries(entry.engines[engine].runs)){
    const tr=node("tr"), q=r.qa_coverage;
    [run,r.episodes,`${r.errors_scheduled} → ${r.errors_triggered}`,`${r.rounds_reached} → ${r.rounds_scored}`,q?`${q.questions.reached} → ${q.questions.scored}`:"—",q?`${q.proactive.reached} → ${q.proactive.scored}`:"—"].forEach(t=>tr.append(node("td",String(t))));
    table.append(tr);
  }
  const wrap=node("div",undefined,"table-wrap");wrap.append(table);details.append(wrap);
}
function render() {
  rows.replaceChildren();details.replaceChildren();
  const selected=entries.filter(e=>`${e.protocol} / ${e.aggregation}`===protocol.value && e.engines[engine]);
  selected.sort((a,b)=>(b.engines[engine].summary.success.mean ?? -1)-(a.engines[engine].summary.success.mean ?? -1));
  status.textContent=selected.length?"":"No reviewed results have been published for this protocol.";
  for(const entry of selected){
    const tr=node("tr"), td=node("td"), button=node("button",entry.assistant.name);
    button.addEventListener("click",()=>detail(entry));td.append(button,node("small",families[entry.assistant.family]||entry.assistant.family));tr.append(td);
    for(const key of ["success","interaction_quality","detection_recall","qa_accuracy"]){
      const score=entry.engines[engine].summary[key];const cell=node("td",pct(score?.mean));
      if(score?.sd !== null && score?.sd !== undefined) cell.title=`Run standard deviation: ${pct(score.sd)}`;
      tr.append(cell);
    }
    const badge=node("td");badge.append(node("span",entry.verification==="sample-audited"?"Sample-audited":"Officially scored","badge"));tr.append(badge);rows.append(tr);
  }
}
document.querySelectorAll("[data-engine]").forEach(button=>button.addEventListener("click",()=>{
  engine=button.dataset.engine;document.querySelectorAll("[data-engine]").forEach(b=>b.setAttribute("aria-pressed",String(b===button)));render();
}));
protocol.addEventListener("change",render);
fetch("feed.json",{cache:"no-store"}).then(r=>{if(!r.ok)throw new Error();return r.json();}).then(data=>{
  if(!Array.isArray(data.entries))throw new Error();entries=data.entries;
  const versions=[...new Set(entries.map(e=>`${e.protocol} / ${e.aggregation}`))];
  if(!versions.length){protocol.append(node("option","No published protocol"));}else for(const v of versions){const opt=node("option",v.split(" / ")[0]);opt.value=v;protocol.append(opt);}
  render();
}).catch(()=>{status.textContent="Reviewed results are not available yet.";document.querySelector("#download").hidden=true;});
