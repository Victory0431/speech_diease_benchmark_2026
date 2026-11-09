/* D12 web console: vanilla JS, no build step, no external assets. */
const S = {
  schema: null,
  catalog: null,
  dataset: null,
  values: {},
  prepared: '',
  split: '',
  tab: 'overview',
  jobId: null,
  pollTimer: null,
  selectedRun: null,
};

const $ = (sel) => document.querySelector(sel);
const esc = (v) => String(v ?? '').replace(/[&<>"']/g, (c) => ({
  '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
}[c]));

function fmt(v, digits = 3) {
  if (v === null || v === undefined || v === '' || Number.isNaN(Number(v))) return '—';
  return Number(v).toFixed(digits);
}

async function api(path, options = {}) {
  const res = await fetch(path, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  });
  const text = await res.text();
  let data = {};
  try { data = text ? JSON.parse(text) : {}; } catch (e) { data = { error: text }; }
  if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
  return data;
}

function toast(message, isError = false) {
  const el = $('#toast');
  el.textContent = message;
  el.classList.toggle('error', isError);
  el.classList.remove('hidden');
  clearTimeout(toast._timer);
  toast._timer = setTimeout(() => el.classList.add('hidden'), 4200);
}

function fieldByKey(key) {
  for (const section of S.schema.sections) {
    const field = section.fields.find((f) => f.key === key);
    if (field) return field;
  }
  return null;
}

function parseValue(field, raw) {
  if (field.type === 'int' || field.type === 'float') {
    if (raw === '' || raw === null) return field.nullable ? null : field.default;
    return Number(raw);
  }
  if (field.type === 'list_int') {
    return String(raw).split(/[,;]/).map((p) => p.trim()).filter(Boolean).map(Number);
  }
  if (field.type === 'bool') return raw === 'true' || raw === true;
  return raw;
}

/* ------------------------------------------------------------ boot */

async function boot() {
  bindGlobal();
  try {
    const [schema, catalog] = await Promise.all([api('/api/schema'), api('/api/catalog')]);
    S.schema = schema;
    S.catalog = catalog;
    $('#workspace-path').textContent = catalog.workspace_root;
    renderDatasetList();
    if (catalog.datasets.length) {
      selectDataset(catalog.datasets[0].id);
    } else {
      $('#dataset-title').textContent = '没有数据集配置';
      $('#panel').innerHTML = '<div class="empty">configs/datasets/ 下没有可用的任务配置。</div>';
    }
  } catch (err) {
    $('#panel').innerHTML = `<div class="empty">加载失败：${esc(err.message)}</div>`;
    toast(err.message, true);
  }
}

function bindGlobal() {
  $('#dataset-list').addEventListener('click', (e) => {
    const li = e.target.closest('li[data-id]');
    if (li) selectDataset(li.dataset.id);
  });
  $('#tabs').addEventListener('click', (e) => {
    const btn = e.target.closest('button[data-tab]');
    if (!btn) return;
    S.tab = btn.dataset.tab;
    document.querySelectorAll('#tabs button').forEach((b) =>
      b.classList.toggle('active', b === btn));
    renderPanel();
  });
  $('#panel').addEventListener('change', onFieldChange);
  $('#panel').addEventListener('input', onFieldChange);
  $('#refresh-btn').onclick = refreshCatalog;
  $('#save-btn').onclick = saveOnly;
  $('#dry-btn').onclick = () => launchRun(true);
  $('#train-btn').onclick = () => launchRun(false);
  $('#log-close').onclick = () => $('#logbar').classList.add('hidden');
  $('#prepared-select').onchange = (e) => {
    S.prepared = e.target.value;
    renderPanel();
  };
  $('#split-select').onchange = (e) => {
    S.split = e.target.value;
    const split = (S.dataset.splits || []).find((s) => s.split_id === S.split);
    if (split) toast(`划分 ${split.split_id} · seed ${split.seed}`);
    renderPanel();
  };
}

function onFieldChange(event) {
  const key = event.target.dataset.key;
  if (!key) return;
  const field = fieldByKey(key);
  if (!field) return;
  try {
    S.values[key] = parseValue(field, event.target.value);
  } catch (err) {
    toast(`参数 ${key} 无效`, true);
  }
  if (key === 'features.n_mfcc') {
    document.querySelectorAll('[data-feature-dim]').forEach((el) => {
      el.textContent = `${2 * Number(S.values[key] || 0)} 维/窗口`;
    });
  }
}

/* ------------------------------------------------------------ dataset */

function renderDatasetList() {
  const list = $('#dataset-list');
  list.innerHTML = S.catalog.datasets.map((ds) => {
    const runs = ds.runs.length;
    const ready = ds.prepared.some((p) => p.status === 'approved');
    return `<li data-id="${esc(ds.id)}" class="${S.dataset?.id === ds.id ? 'active' : ''}">
      <span class="name">${esc(ds.id)}</span>
      <span class="meta">${esc(ds.adapter || '—')} · ${esc(ds.grouping)} · ${runs} runs${
        ready ? '' : ' · 未准入'}</span>
    </li>`;
  }).join('');
}

function selectDataset(id) {
  S.dataset = S.catalog.datasets.find((d) => d.id === id) || null;
  if (!S.dataset) return;
  S.values = { ...(S.dataset.defaults || {}) };
  const approved = S.dataset.prepared.filter((p) => p.status === 'approved');
  S.prepared = S.dataset.defaults.__prepared_id
    || (approved[0] ? approved[0].job_id : '');
  S.split = S.dataset.defaults.__split_id
    || (S.dataset.splits[0] ? S.dataset.splits[0].split_id : '');
  S.selectedRun = S.dataset.runs[0] ? S.dataset.runs[0].run_id : null;
  $('#experiment-id').value = '';
  renderDatasetList();
  renderHeader();
  renderBindSelects();
  renderPanel();
}

function renderHeader() {
  const ds = S.dataset;
  $('#dataset-title').textContent = `${ds.id} · ${ds.task_id}`;
  const runs = ds.runs.length;
  const approved = ds.prepared.filter((p) => p.status === 'approved').length;
  $('#dataset-badges').innerHTML = [
    `<span class="badge">adapter ${esc(ds.adapter || '—')}</span>`,
    `<span class="badge ${ds.grouping === 'file' ? 'warn' : 'ok'}">grouping ${esc(ds.grouping)}</span>`,
    `<span class="badge gray">主单位 ${esc(ds.defaults['evaluation.unit'] || '—')}</span>`,
    `<span class="badge gray">${ds.splits.length} 个划分</span>`,
    `<span class="badge ${approved ? 'ok' : 'warn'}">${approved} 个已准入</span>`,
    `<span class="badge gray">${runs} 个 run</span>`,
  ].join('');
}

function renderBindSelects() {
  const prepared = S.dataset.prepared.filter((p) => p.status === 'approved');
  const pSel = $('#prepared-select');
  pSel.innerHTML = prepared.length
    ? prepared.map((p) => `<option value="${esc(p.job_id)}">${esc(p.job_id)} · ${
        p.counts?.n_manifest ?? '?'} 条</option>`).join('')
    : '<option value="">没有 approved 的 prepared 作业</option>';
  pSel.value = S.prepared;
  const sSel = $('#split-select');
  sSel.innerHTML = S.dataset.splits.length
    ? S.dataset.splits.map((s) => `<option value="${esc(s.split_id)}">${esc(s.split_id)} · seed ${
        s.seed}</option>`).join('')
    : '<option value="">没有划分</option>';
  sSel.value = S.split;
}

/* ------------------------------------------------------------ panel */

function renderPanel() {
  if (!S.dataset) return;
  const panels = {
    overview: renderOverview,
    data: renderData,
    split: renderSplit,
    model: renderModel,
    results: renderResults,
  };
  $('#panel').innerHTML = (panels[S.tab] || renderOverview)();
  if (S.tab === 'results') renderGallery();
}

function statusDot(ok, warn = false) {
  return `<span class="dot ${ok ? 'ok' : warn ? 'warn' : 'bad'}"></span>`;
}

function renderOverview() {
  const ds = S.dataset;
  const scans = ds.scans.length;
  const approved = ds.prepared.filter((p) => p.status === 'approved');
  const lastScan = ds.scans[0];
  const lastPrepared = approved[0];
  const lastSplit = ds.splits[0];
  const lastRun = ds.runs[0];
  const counts = lastPrepared?.counts || {};
  const dim = 2 * Number(S.values['features.n_mfcc'] || 0);

  const cards = [
    ['数据集', ds.id, 'small'],
    ['适配器', ds.adapter || '—', 'small'],
    ['隔离单位', ds.grouping, ''],
    ['说话人 / 组', `${counts.n_subjects ?? '—'} / ${counts.n_groups ?? '—'}`, ''],
    ['可用样本', counts.n_manifest ?? '—', ''],
    ['特征维度', `${dim} 维/窗口`, ''],
    ['划分 / run', `${ds.splits.length} / ${ds.runs.length}`, ''],
  ].map(([k, v, cls]) => `<div class="card"><span class="k">${k}</span>
      <span class="v ${cls}">${esc(v)}</span></div>`).join('');

  const check = (ok, label, detail, command) => `<div class="check">
      <div>${statusDot(ok)}</div>
      <div class="label">${label}</div>
      <div class="detail">${detail}${command ? `<br><code>${esc(command)}</code>` : ''}</div>
    </div>`;

  const checklist = [
    check(scans > 0, '扫描与统计',
      scans ? `${scans} 个扫描 · 最新 ${esc(lastScan.scan_id)}（${esc(lastScan.mode)}/${esc(lastScan.status)}）`
        : '还没有完整扫描，建议 full 模式。',
      `sddbench stats --dataset ${ds.id} --mode full --paths configs/paths.local.yaml`),
    check(approved.length > 0, '标签准入',
      approved.length ? `${approved.length} 个 approved · 最新 ${esc(lastPrepared.job_id)}`
        : '缺少 approved 的 prepared 作业。',
      `sddbench prepare --dataset ${ds.id} --source <scan_id> --paths configs/paths.local.yaml`),
    check(ds.splits.length > 0, '数据划分',
      ds.splits.length ? `${ds.splits.length} 个划分 · 最新 seed ${esc(lastSplit.seed)}`
        : '还没有划分。',
      `sddbench split --dataset ${ds.id} --prepared <prepared_id> --seed 0 --paths configs/paths.local.yaml`),
    check(true, '特征缓存', '缓存键 = 特征配置 + 采样率 + 窗口指纹；改数据/特征参数后必须重算。',
      `sddbench extract-features --dataset ${ds.id} --source <scan_id> --preprocess <yaml> --features <yaml> --paths configs/paths.local.yaml`),
    check(ds.runs.length > 0, '训练与结果',
      ds.runs.length ? `${ds.runs.length} 个 run · 最新 bAcc ${fmt(lastRun.balanced_accuracy)}（${esc(lastRun.primary_unit)}）`
        : '还没有训练结果。',
      '在“模型与训练”页点“启动训练”，或命令行 sddbench run --config <yaml>'),
  ].join('');

  const configRows = Object.entries(ds.raw_config || {}).map(([k, v]) =>
    `<tr><td class="mono">${esc(k)}</td><td class="mono">${
      esc(typeof v === 'object' ? JSON.stringify(v, null, 0) : v)}</td></tr>`).join('');

  const runRows = ds.runs.slice(0, 6).map((r) => `<tr>
      <td class="mono">${esc(r.run_id)}</td>
      <td class="mono">${esc(r.experiment_id)}</td>
      <td>${esc(r.seed)}</td>
      <td>${esc(r.primary_unit)}</td>
      <td class="num">${fmt(r.balanced_accuracy)}</td>
      <td class="num">${fmt(r.auroc)}</td>
      <td>${esc(r.best_epoch ?? '—')}</td>
    </tr>`).join('');

  return `
    <div class="cards">${cards}</div>
    <div class="block">
      <div class="block-head"><h2>流水线状态</h2>
        <span class="desc">每一步都必须有显式产物；红色表示还没就绪。</span></div>
      <div class="block-body"><div class="checklist">${checklist}</div></div>
    </div>
    <div class="block">
      <div class="block-head"><h2>数据集配置</h2>
        <span class="desc">${esc(ds.config_path)}</span></div>
      <div class="block-body tight"><div class="table-wrap"><table>
        <thead><tr><th>键</th><th>值</th></tr></thead>
        <tbody>${configRows || '<tr><td colspan="2">无</td></tr>'}</tbody>
      </table></div></div>
    </div>
    <div class="block">
      <div class="block-head"><h2>最近 run</h2>
        <span class="desc">主单位指标，完整榜单见“结果”页。</span></div>
      <div class="block-body tight"><div class="table-wrap"><table>
        <thead><tr><th>run_id</th><th>experiment</th><th>seed</th><th>单位</th>
          <th>bAcc</th><th>AUROC</th><th>best epoch</th></tr></thead>
        <tbody>${runRows || '<tr><td colspan="7">还没有 run</td></tr>'}</tbody>
      </table></div></div>
    </div>`;
}

function renderSection(section) {
  const impact = S.schema.impact_labels[section.fields[0]?.impact] || '';
  const fields = section.fields.map((f) => renderField(f)).join('');
  return `<div class="block">
    <div class="block-head"><h2>${esc(section.title)}</h2>
      <span class="desc">${esc(section.description || '')}</span></div>
    <div class="block-body">
      ${impact ? `<div class="impact">改动本组参数：${esc(impact)}。</div>` : ''}
      <div class="form-grid">${fields}</div>
    </div>
  </div>`;
}

function renderField(f) {
  const value = S.values[f.key];
  let control = '';
  if (f.type === 'select') {
    control = `<select data-key="${esc(f.key)}">${(f.options || []).map((opt) =>
      `<option value="${esc(opt)}" ${String(value) === String(opt) ? 'selected' : ''}>${esc(opt)}</option>`
    ).join('')}</select>`;
  } else if (f.type === 'bool') {
    control = `<select data-key="${esc(f.key)}">
      <option value="true" ${value ? 'selected' : ''}>true</option>
      <option value="false" ${!value ? 'selected' : ''}>false</option>
    </select>`;
  } else if (f.type === 'list_int') {
    control = `<input class="mono" data-key="${esc(f.key)}" value="${esc(
      Array.isArray(value) ? value.join(',') : value)}" placeholder="128 或 64,32">`;
  } else {
    const kind = (f.type === 'int' || f.type === 'float') ? 'number' : 'text';
    const step = f.step ? `step="${f.step}"` : (f.type === 'int' ? 'step="1"' : 'step="any"');
    control = `<input type="${kind}" data-key="${esc(f.key)}" value="${
      value === null || value === undefined ? '' : esc(value)}" ${step}
      ${f.min !== null && f.min !== undefined ? `min="${f.min}"` : ''}
      ${f.max !== null && f.max !== undefined ? `max="${f.max}"` : ''}>`;
  }
  const unit = f.key === 'features.n_mfcc'
    ? `<span class="help" data-feature-dim>${2 * Number(S.values['features.n_mfcc'] || 0)} 维/窗口</span>` : '';
  return `<div class="field">
    <label>${esc(f.label)}</label>
    ${control}
    ${f.help ? `<span class="help">${esc(f.help)}</span>` : ''}
    ${unit}
  </div>`;
}

function sectionById(id) {
  return S.schema.sections.find((s) => s.id === id);
}

function renderData() {
  return renderSection(sectionById('preprocess')) + renderSection(sectionById('features'));
}

function renderSplit() {
  const splitSection = sectionById('split');
  const rows = S.dataset.splits.map((s) => {
    const bySplit = s.summary?.by_split || {};
    const chips = Object.entries(bySplit).map(([name, info]) =>
      `<span class="chip">${esc(name)} ${esc(info.n_samples ?? '?')} 条</span>`).join('');
    return `<tr>
      <td class="mono">${esc(s.split_id)}</td>
      <td>${esc(s.seed)}</td>
      <td>${esc(s.prepared_id)}</td>
      <td>${esc(s.summary?.n_samples ?? '—')}</td>
      <td>${esc(s.summary?.n_groups ?? s.n_groups ?? '—')}</td>
      <td><div class="split-line">${chips || '—'}</div></td>
      <td>${S.split === s.split_id ? '<span class="badge ok">当前</span>' : ''}</td>
    </tr>`;
  }).join('');
  return `
    ${renderSection(splitSection)}
    <div class="block">
      <div class="block-head"><h2>已有划分</h2>
        <button class="btn" id="split-btn">用上面的 seed 生成新划分</button></div>
      <div class="block-body tight"><div class="table-wrap"><table>
        <thead><tr><th>split_id</th><th>seed</th><th>prepared</th><th>样本</th>
          <th>组数</th><th>分布</th><th></th></tr></thead>
        <tbody>${rows || '<tr><td colspan="7">还没有划分</td></tr>'}</tbody>
      </table></div></div>
    </div>`;
}

function renderModel() {
  return renderSection(sectionById('model'))
    + renderSection(sectionById('training'))
    + renderSection(sectionById('evaluation'));
}

function table(columns, rows, emptyText) {
  const head = columns.map((c) => `<th>${esc(c.label)}</th>`).join('');
  const body = rows.length ? rows.map((row) => `<tr>${columns.map((c) => {
    const value = c.render ? c.render(row) : row[c.key];
    return `<td class="${c.cls || ''}">${c.raw ? value : esc(value)}</td>`;
  }).join('')}</tr>`).join('')
    : `<tr><td colspan="${columns.length}">${emptyText}</td></tr>`;
  return `<div class="table-wrap"><table><thead><tr>${head}</tr></thead>
    <tbody>${body}</tbody></table></div>`;
}

function renderResults() {
  const results = S.catalog.results || {};
  const lb = results.leaderboard || [];
  const agg = results.aggregates || [];
  const splits = results.split_sensitivity || [];
  const runs = S.dataset.runs;

  const cards = [
    ['本库 run', runs.length],
    ['最佳 bAcc', runs.length ? fmt(Math.max(...runs.map((r) => Number(r.balanced_accuracy) || 0))) : '—'],
    ['榜单行', lb.length],
    ['聚合组', agg.length],
  ].map(([k, v]) => `<div class="card"><span class="k">${k}</span>
    <span class="v">${esc(v)}</span></div>`).join('');

  const lbTable = table([
    { label: 'dataset', key: 'dataset', cls: 'mono' },
    { label: 'experiment', key: 'experiment_id', cls: 'mono' },
    { label: 'unit', key: 'primary_unit' },
    { label: 'seed', key: 'seed' },
    { label: 'bAcc', key: 'primary_balanced_accuracy',
      render: (r) => fmt(r.primary_balanced_accuracy ?? r.balanced_accuracy), cls: 'num' },
    { label: 'AUROC', key: 'primary_auroc',
      render: (r) => fmt(r.primary_auroc ?? r.auroc), cls: 'num' },
    { label: 'split', key: 'split_id', cls: 'mono' },
  ], lb, '还没有归档结果，先运行 sddbench publish');

  const aggTable = table([
    { label: 'dataset', key: 'dataset' },
    { label: 'experiment', key: 'experiment_id', cls: 'mono' },
    { label: 'unit', key: 'primary_unit' },
    { label: 'dropout', key: 'dropout' },
    { label: 'runs', key: 'n_runs', cls: 'num' },
    { label: 'bAcc mean', key: 'balanced_accuracy_mean',
      render: (r) => fmt(r.balanced_accuracy_mean), cls: 'num' },
    { label: '95% CI', key: 'balanced_accuracy_ci_low',
      render: (r) => `[${fmt(r.balanced_accuracy_ci_low)}, ${fmt(r.balanced_accuracy_ci_high)}]`,
      cls: 'num' },
  ], agg, '还没有 aggregate 结果');

  const splitTable = table([
    { label: 'dataset', key: 'dataset' },
    { label: 'experiment', key: 'experiment_id', cls: 'mono' },
    { label: 'unit', key: 'primary_unit' },
    { label: 'split', key: 'split_id', cls: 'mono' },
    { label: 'bAcc mean', key: 'balanced_accuracy_mean',
      render: (r) => fmt(r.balanced_accuracy_mean), cls: 'num' },
    { label: 'across-split mean', key: 'across_splits_mean',
      render: (r) => fmt(r.across_splits_mean), cls: 'num' },
    { label: 'range', key: 'across_splits_range',
      render: (r) => fmt(r.across_splits_range), cls: 'num' },
  ], splits, '还没有换划分敏感性结果');

  const runOptions = runs.map((r) => `<option value="${esc(r.run_id)}" ${
    S.selectedRun === r.run_id ? 'selected' : ''}>${esc(r.run_id)} · ${esc(r.primary_unit)} · bAcc ${
    fmt(r.balanced_accuracy)}</option>`).join('');

  return `
    <div class="cards">${cards}</div>
    <div class="block"><div class="block-head"><h2>跨数据集榜单</h2>
      <span class="desc">不同 primary_unit 不直接比较。</span></div>
      <div class="block-body tight">${lbTable}</div></div>
    <div class="block"><div class="block-head"><h2>多 seed 聚合</h2>
      <span class="desc">来自 results_summary/aggregates.csv</span></div>
      <div class="block-body tight">${aggTable}</div></div>
    <div class="block"><div class="block-head"><h2>换划分敏感性</h2>
      <span class="desc">来自 results_summary/split_sensitivity.csv</span></div>
      <div class="block-body tight">${splitTable}</div></div>
    <div class="block"><div class="block-head"><h2>单次 run 图表</h2>
      <select id="run-gallery-select">${runOptions || '<option value="">没有 run</option>'}</select>
      </div>
      <div class="block-body"><div class="gallery" id="run-gallery"></div></div></div>`;
}

function renderGallery() {
  const select = $('#run-gallery-select');
  if (!select) return;
  select.onchange = (e) => { S.selectedRun = e.target.value; renderGallery(); };
  const run = S.dataset.runs.find((r) => r.run_id === S.selectedRun)
    || S.dataset.runs[0];
  if (!run) { $('#run-gallery').innerHTML = '<div class="empty">没有 run</div>'; return; }
  S.selectedRun = run.run_id;
  const figures = run.figures && run.figures.length ? run.figures : [
    'confusion_test.png', 'roc_test.png', 'scores_test.png', 'learning_curve.png'];
  $('#run-gallery').innerHTML = figures.map((name) => `
    <figure>
      <img src="/media/run/${encodeURIComponent(run.run_id)}/${encodeURIComponent(name)}"
           alt="${esc(name)}" loading="lazy"
           onerror="this.outerHTML='&lt;div class=&quot;missing&quot;&gt;暂无图像&lt;/div&gt;'">
      <figcaption>${esc(name)}</figcaption>
    </figure>`).join('');
}

/* ------------------------------------------------------------ actions */

function buildBody() {
  return {
    dataset: S.dataset.id,
    prepared_id: S.prepared,
    split_id: S.split,
    experiment_id: $('#experiment-id').value.trim(),
    values: S.values,
  };
}

async function saveOnly() {
  if (!checkBinding()) return;
  try {
    const res = await api('/api/experiment', {
      method: 'POST',
      body: JSON.stringify(buildBody()),
    });
    toast(`配置已写入 ${res.paths.experiment} · 特征 ${res.feature_dim} 维`);
  } catch (err) {
    toast(err.message, true);
  }
}

async function launchRun(dryRun) {
  if (!checkBinding()) return;
  try {
    const res = await api('/api/run', {
      method: 'POST',
      body: JSON.stringify({ ...buildBody(), dry_run: dryRun }),
    });
    const saved = res.saved;
    toast(`配置 ${saved.paths.experiment} · 特征 ${saved.feature_dim} 维 · ${
      dryRun ? '试运行' : '开始训练'}`);
    openLog(res.job);
    pollJob(res.job.job_id);
  } catch (err) {
    toast(err.message, true);
  }
}

function checkBinding() {
  if (!S.dataset) return false;
  if (!S.prepared) { toast('请先选择 approved 的 prepared 作业', true); return false; }
  if (!S.split) { toast('请先在“划分”页生成或选择划分', true); return false; }
  return true;
}

async function createSplit() {
  if (!S.prepared) { toast('请先选择 prepared 作业', true); return; }
  const btn = $('#split-btn');
  if (btn) btn.disabled = true;
  try {
    const res = await api('/api/split', {
      method: 'POST',
      body: JSON.stringify({
        dataset: S.dataset.id,
        prepared_id: S.prepared,
        seed: S.values['split.seed'],
        train_frac: S.values['split.train_frac'],
        val_frac: S.values['split.val_frac'],
        test_frac: S.values['split.test_frac'],
      }),
    });
    S.split = res.split_id;
    toast(`新划分：${res.split_id}`);
    await refreshCatalog(false);
  } catch (err) {
    toast(err.message, true);
  } finally {
    if (btn) btn.disabled = false;
  }
}

async function refreshCatalog(resetValues = false) {
  try {
    S.catalog = await api('/api/catalog');
    const current = S.catalog.datasets.find((d) => d.id === S.dataset?.id) || S.catalog.datasets[0];
    if (!current) return;
    const keepValues = !resetValues && S.dataset && current.id === S.dataset.id
      ? S.values : null;
    S.dataset = current;
    S.values = keepValues || { ...current.defaults };
    const approved = current.prepared.filter((p) => p.status === 'approved');
    if (!approved.some((p) => p.job_id === S.prepared)) {
      S.prepared = current.defaults.__prepared_id || (approved[0]?.job_id || '');
    }
    if (!current.splits.some((s) => s.split_id === S.split)) {
      S.split = current.defaults.__split_id || (current.splits[0]?.split_id || '');
    }
    renderDatasetList();
    renderHeader();
    renderBindSelects();
    renderPanel();
  } catch (err) {
    toast(err.message, true);
  }
}

function openLog(job) {
  $('#logbar').classList.remove('hidden');
  $('#log-title').textContent = `${job.kind} · ${job.job_id}`;
  setLogStatus(job.status);
  $('#log-body').textContent = '等待日志…';
}

function setLogStatus(status) {
  const el = $('#log-status');
  el.textContent = status;
  el.className = `pill ${status}`;
}

async function pollJob(jobId, attempt = 0) {
  clearTimeout(S.pollTimer);
  try {
    const job = await api(`/api/jobs/${jobId}`);
    $('#log-body').textContent = job.log || '（空）';
    $('#log-body').scrollTop = $('#log-body').scrollHeight;
    setLogStatus(job.status);
    if (job.status === 'running') {
      S.pollTimer = setTimeout(() => pollJob(jobId, attempt + 1), 1200);
      return;
    }
    if (job.status === 'completed') {
      toast(job.run_id ? `训练完成：${job.run_id}` : '试运行通过');
      S.selectedRun = job.run_id || S.selectedRun;
      await refreshCatalog(false);
      if (job.run_id) {
        S.tab = 'results';
        document.querySelectorAll('#tabs button').forEach((b) =>
          b.classList.toggle('active', b.dataset.tab === 'results'));
        renderPanel();
      }
    } else {
      toast(`任务失败（exit ${job.returncode}），见日志`, true);
    }
  } catch (err) {
    toast(err.message, true);
  }
}

document.addEventListener('DOMContentLoaded', boot);
document.addEventListener('click', (e) => {
  if (e.target.id === 'split-btn') createSplit();
});
