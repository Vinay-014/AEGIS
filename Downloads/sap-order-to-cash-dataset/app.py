import os
import re
import json
import zipfile
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Set

import streamlit as st
import streamlit.components.v1 as components

import pandas as pd
import networkx as nx
from pyvis.network import Network


BASE_DIR = Path(__file__).resolve().parent


def _val(v: Any) -> Optional[str]:
    if v is None:
        return None
    s = str(v).strip()
    if s in ("", "nan", "None", "NaT"):
        return None
    if "T00:00:00" in s:
        s = s.split("T")[0]
    return s


def _make_tooltip_data(
    entity_type: str, fields_dict: Dict[str, Any], connections: int = 0
) -> str:
    rows = []
    for label, raw in fields_dict.items():
        v = _val(raw)
        if v is not None:
            rows.append({"k": label, "v": v})
    data = {"type": entity_type, "rows": rows, "connections": connections}
    return json.dumps(data, ensure_ascii=True)


@st.cache_resource(show_spinner="Loading and building graph (one-time)…")
def load_graph() -> nx.DiGraph:
    zip_path = BASE_DIR / "sap-order-to-cash-dataset.zip"
    extract_dir = BASE_DIR / "sap-dataset"
    data_dir = BASE_DIR / "sap-o2c-data"

    if not data_dir.exists():
        if zip_path.exists():
            extract_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(extract_dir)
        else:
            raise FileNotFoundError(
                f"Dataset not found. Expected {data_dir} or {zip_path}."
            )

    def load_jsonl_data(subdirectory_name: str) -> pd.DataFrame:
        full_path = data_dir / subdirectory_name
        if not full_path.exists():
            return pd.DataFrame()
        jsonl_files = [p for p in full_path.iterdir() if p.name.endswith(".jsonl")]
        if not jsonl_files:
            return pd.DataFrame()
        dfs = []
        for f in jsonl_files:
            try:
                dfs.append(pd.read_json(f, lines=True))
            except Exception:
                continue
        if not dfs:
            return pd.DataFrame()
        return pd.concat(dfs, ignore_index=True)

    df_orders  = load_jsonl_data("sales_order_headers")
    df_items   = load_jsonl_data("sales_order_items")
    df_billing = load_jsonl_data("billing_document_headers")
    df_journal = load_jsonl_data("journal_entry_items_accounts_receivable")

    if df_orders.empty or df_items.empty:
        raise ValueError("Core datasets empty. Check extraction.")

    G = nx.DiGraph()
    COLORS = {
        "Customer":     "#1e88e5",
        "Order":        "#ff7043",
        "LineItem":     "#66bb6a",
        "Product":      "#ffee58",
        "Invoice":      "#ab47bc",
        "JournalEntry": "#26a69a",
    }

    if "soldToParty" in df_orders.columns:
        for cust_id_val, grp in df_orders.groupby("soldToParty"):
            cid = f"Cust_{cust_id_val}"
            G.add_node(cid,
                label="Customer", type="Customer", color=COLORS["Customer"],
                title=_make_tooltip_data("Customer", {
                    "Customer ID":  cust_id_val,
                    "Total Orders": len(grp),
                }, connections=len(grp)),
            )

    for _, r in df_orders.iterrows():
        if "salesOrder" not in r or "soldToParty" not in r:
            continue
        oid = f"Order_{r['salesOrder']}"
        item_cnt = (
            len(df_items[df_items["salesOrder"] == r["salesOrder"]])
            if not df_items.empty and "salesOrder" in df_items.columns else 0
        )
        G.add_node(oid,
            label=f"Order {r.get('salesOrder')}",
            type="Order", color=COLORS["Order"],
            amount=r.get("totalNetAmount"),
            title=_make_tooltip_data("Sales Order", {
                "Sales Order":  r.get("salesOrder"),
                "Company Code": r.get("companyCode"),
                "Sales Org":    r.get("salesOrganization"),
                "Order Type":   r.get("salesOrderType"),
                "Net Amount":   f"{r.get('totalNetAmount','')} {r.get('transactionCurrency','')}",
                "Created On":   r.get("creationDate"),
                "Customer":     r.get("soldToParty"),
            }, connections=item_cnt + 1),
        )
        G.add_edge(f"Cust_{r['soldToParty']}", oid, relationship="PLACED")

    product_usage: Dict[str, int] = {}
    if not df_items.empty and "material" in df_items.columns:
        product_usage = df_items["material"].value_counts().to_dict()

    for _, r in df_items.iterrows():
        if "salesOrder" not in r or "salesOrderItem" not in r or "material" not in r:
            continue
        iid = f"Item_{r['salesOrder']}_{r['salesOrderItem']}"
        pid = f"Prod_{r['material']}"
        G.add_node(iid,
            label="Item", type="LineItem", color=COLORS["LineItem"],
            title=_make_tooltip_data("Order Item", {
                "Sales Order": r.get("salesOrder"),
                "Item No":     r.get("salesOrderItem"),
                "Material":    r.get("material"),
                "Description": r.get("materialDescription", r.get("material")),
                "Quantity":    f"{r.get('requestedQuantity','')} {r.get('requestedQuantityUnit','')}",
                "Net Price":   r.get("netPrice"),
                "Plant":       r.get("plant"),
            }, connections=2),
        )
        G.add_node(pid,
            label=f"Prod {r.get('material')}",
            type="Product", color=COLORS["Product"],
            title=_make_tooltip_data("Product", {
                "Material":    r.get("material"),
                "Description": r.get("materialDescription", r.get("material")),
                "Mat Type":    r.get("materialType"),
                "Mat Group":   r.get("materialGroup"),
                "Base Unit":   r.get("baseUnitOfMeasure"),
                "Used in":     f"{product_usage.get(r['material'], 1)} orders",
            }, connections=product_usage.get(r["material"], 1)),
        )
        G.add_edge(f"Order_{r['salesOrder']}", iid, relationship="CONTAINS")
        G.add_edge(iid, pid, relationship="REFERENCES_PRODUCT")

    if not df_billing.empty and "billingDocument" in df_billing.columns:
        for _, r in df_billing.iterrows():
            if "billingDocument" not in r or "soldToParty" not in r:
                continue
            bid = f"Bill_{r['billingDocument']}"
            je_c = 0
            if not df_journal.empty and "referenceDocument" in df_journal.columns:
                je_c = len(df_journal[df_journal["referenceDocument"] == r["billingDocument"]])
            G.add_node(bid,
                label=f"Inv {r.get('billingDocument')}",
                type="Invoice", color=COLORS["Invoice"],
                doc_id=r.get("billingDocument"),
                title=_make_tooltip_data("Billing Document", {
                    "Billing Doc":  r.get("billingDocument"),
                    "Type":         r.get("billingType"),
                    "Company Code": r.get("companyCode"),
                    "Billing Date": r.get("billingDate"),
                    "Net Amount":   r.get("totalNetAmount"),
                    "Tax Amount":   r.get("totalTaxAmount"),
                    "Currency":     r.get("transactionCurrency"),
                    "Customer":     r.get("soldToParty"),
                    "Payer":        r.get("payerParty"),
                }, connections=1 + je_c),
            )
            cust_id = f"Cust_{r['soldToParty']}"
            if cust_id in G:
                G.add_edge(cust_id, bid, relationship="BILLED_TO")

    if not df_journal.empty and "accountingDocument" in df_journal.columns:
        for _, r in df_journal.iterrows():
            if "accountingDocument" not in r or "referenceDocument" not in r:
                continue
            jid = f"JE_{r['accountingDocument']}"
            G.add_node(jid,
                label=f"JE {r.get('accountingDocument')}",
                type="JournalEntry", color=COLORS["JournalEntry"],
                je_id=r.get("accountingDocument"),
                title=_make_tooltip_data("Journal Entry", {
                    "CompanyCode":                 r.get("companyCode"),
                    "FiscalYear":                  r.get("fiscalYear"),
                    "AccountingDocument":          r.get("accountingDocument"),
                    "GlAccount":                   r.get("glAccount"),
                    "ReferenceDocument":           r.get("referenceDocument"),
                    "CostCenter":                  r.get("costCenter"),
                    "ProfitCenter":                r.get("profitCenter"),
                    "TransactionCurrency":         r.get("transactionCurrency"),
                    "AmountInTransactionCurrency": r.get("amountInTransactionCurrency"),
                    "CompanyCodeCurrency":         r.get("companyCodeCurrency"),
                    "AmountInCompanyCodeCurrency": r.get("amountInCompanyCodeCurrency"),
                    "PostingDate":                 r.get("postingDate"),
                    "DocumentDate":                r.get("documentDate"),
                    "AccountingDocumentType":      r.get("accountingDocumentType"),
                    "AccountingDocumentItem":      r.get("accountingDocumentItem"),
                }, connections=2),
            )
            bill_ref = f"Bill_{r['referenceDocument']}"
            if bill_ref in G:
                G.add_edge(bill_ref, jid, relationship="POSTED_AS")

    return G


def format_response(sections: List[Tuple[str, Any]]) -> str:
    out: List[str] = []
    for kind, content in sections:
        if kind == "heading":
            out.append(
                '<div style="font-weight:700;color:#f8fafc;font-size:12px;'
                'letter-spacing:0.02em;margin-bottom:6px;padding-bottom:4px;'
                'border-bottom:1px solid #334155;">'
                f'{content}</div>'
            )
        elif kind == "row":
            label, val = content
            out.append(
                '<div style="display:flex;gap:6px;padding:2px 0;">'
                f'<span style="color:#94a3b8;min-width:80px;flex-shrink:0;font-size:11px;">{label}</span>'
                f'<span style="color:#f8fafc;font-weight:600;font-size:11px;">{val}</span>'
                "</div>"
            )
        elif kind == "list":
            rows = "".join(
                '<div style="display:flex;gap:6px;padding:2px 0;">'
                f'<span style="color:#94a3b8;font-size:11px;min-width:18px;">{i+1}.</span>'
                f'<span style="color:#f8fafc;font-size:11px;">{item}</span>'
                '</div>'
                for i, item in enumerate(content)
            )
            out.append(rows)
        elif kind == "status":
            out.append(
                '<div style="margin-top:8px;padding:6px 10px;background:#0f172a;'
                'border:1px solid #3b82f6;border-radius:6px;color:#f8fafc;'
                'font-size:10.5px;font-weight:600;">'
                f'<span style="color:#ef4444;margin-right:6px;">&#9679;</span>{content}</div>'
            )
        elif kind == "divider":
            out.append('<div style="border-top:1px solid #334155;margin:6px 0;"></div>')
        elif kind == "body":
            out.append(
                f'<div style="color:#f8fafc;font-size:11.5px;line-height:1.6;">{content}</div>'
            )
        elif kind == "error":
            out.append(
                f'<div style="color:#f87171;font-size:11px;">{content}</div>'
            )
    return "".join(out)


def query_graph_ai(
    user_query: str, graph_instance: nx.DiGraph
) -> Tuple[bool, str, List[str]]:
    highlighted_nodes: List[str] = []
    q = (user_query or "").lower()

    o2c_keywords = [
        "order", "customer", "billing", "invoice", "product", "item",
        "sale", "sold", "qty", "amount", "journal", "entry", "document",
        "flow", "trace", "incomplete", "delivery", "payment", "broken",
        "pending", "billed", "associated", "highest", "number",
    ]
    if not any(k in q for k in o2c_keywords):
        return False, format_response([
            ("body", "This system answers questions related to the "
                     "<strong>SAP Order-to-Cash dataset</strong> only."),
            ("divider", ""),
            ("body", 'Try: <em>"Trace order 740506"</em> or <em>"Show broken flows"</em>'),
        ]), []

    doc_match = re.search(r"\b(\d{6,})\b", q)
    doc_id = doc_match.group(1) if doc_match else None

    if "product" in q and ("highest" in q or "most" in q or "top" in q):
        counts: Dict[str, int] = {}
        for node, attrs in graph_instance.nodes(data=True):
            if attrs.get("type") == "Product":
                related: Set[str] = set()
                for item in graph_instance.predecessors(node):
                    for order in graph_instance.predecessors(item):
                        for cust in graph_instance.predecessors(order):
                            for inv in graph_instance.successors(cust):
                                if "Bill" in inv:
                                    related.add(inv)
                counts[node] = len(related)
        top5 = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:5]
        highlighted_nodes = [p[0] for p in top5]
        return True, format_response([
            ("heading", "Top Products — Billing Document Association"),
            ("list", [
                f'<strong style="color:#f8fafc;">{pid.replace("Prod_","")}</strong>'
                f'<span style="color:#94a3b8;"> &nbsp;·&nbsp; {cnt} billing documents</span>'
                for pid, cnt in top5
            ]),
            ("status", f"{len(highlighted_nodes)} product nodes highlighted in RED"),
        ]), highlighted_nodes

    if any(k in q for k in ["incomplete", "broken", "not billed", "pending"]):
        orders = [n for n, d in graph_instance.nodes(data=True) if d.get("type") == "Order"]
        incomplete: List[str] = []
        for order in orders[:300]:
            custs = [n for n in graph_instance.predecessors(order) if "Cust" in n]
            if custs and not [n for n in graph_instance.successors(custs[0]) if "Bill" in n]:
                incomplete.append(order)
        if incomplete:
            highlighted_nodes = incomplete[:15]
            return True, format_response([
                ("heading", "Incomplete Flow Analysis"),
                ("row", ("Orders without billing", str(len(incomplete)))),
                ("divider", ""),
                ("body", "<strong>Sample orders:</strong>"),
                ("list", [
                    f'<strong style="color:#f8fafc;">{o.replace("Order_","")}</strong>'
                    f'<span style="color:#94a3b8;"> &nbsp;·&nbsp; '
                    f'Amt: {graph_instance.nodes[o].get("amount","N/A")}</span>'
                    for o in incomplete[:10]
                ]),
                ("status", f"{len(highlighted_nodes)} order nodes highlighted in RED"),
            ]), highlighted_nodes
        return False, format_response([
            ("body", "All sampled orders have associated billing documents.")
        ]), []

    if ("trace" in q or "flow" in q) and ("order" in q or doc_id):
        order_id = doc_id
        if not order_id:
            m = re.search(r"order[_\s]*(\d+)", q)
            order_id = m.group(1) if m else None
        if order_id and f"Order_{order_id}" in graph_instance:
            oid = f"Order_{order_id}"
            highlighted_nodes = [oid]
            customers = [n for n in graph_instance.predecessors(oid) if "Cust" in n]
            cust = customers[0] if customers else None
            if cust:
                highlighted_nodes.append(cust)
            items = list(graph_instance.successors(oid))
            highlighted_nodes.extend(items[:10])
            products: Set[str] = set()
            for item in items[:10]:
                products.update(graph_instance.successors(item))
            highlighted_nodes.extend(list(products)[:5])
            invoices = [n for n in graph_instance.successors(cust) if "Bill" in n] if cust else []
            highlighted_nodes.extend(invoices[:5])
            jes: List[str] = []
            for inv in invoices[:5]:
                jes.extend(n for n in graph_instance.successors(inv) if "JE" in n)
            highlighted_nodes.extend(jes[:5])
            attrs = graph_instance.nodes[oid]
            sections: List[Tuple[str, Any]] = [
                ("heading", f"Flow Trace — Order {order_id}"),
                ("row", ("Net Amount",      str(attrs.get("amount", "N/A")))),
                ("row", ("Customer",        cust.replace("Cust_", "") if cust else "N/A")),
                ("divider", ""),
                ("row", ("Line Items",      str(len(items)))),
                ("row", ("Products",        str(len(products)))),
                ("row", ("Invoices",        str(len(invoices)))),
                ("row", ("Journal Entries", str(len(jes)))),
            ]
            if jes:
                sections.append(("row", ("JE Numbers",
                    ", ".join(j.replace("JE_", "") for j in jes[:3]))))
            sections.append(("status", f"{len(highlighted_nodes)} nodes highlighted in RED"))
            return True, format_response(sections), highlighted_nodes
        return False, format_response([
            ("error", f"Order {order_id} not found in dataset.")
        ]), []

    if any(k in q for k in ["count", "how many", "total"]):
        for kw, typ in [
            ("order", "Order"), ("customer", "Customer"),
            ("invoice", "Invoice"), ("billing", "Invoice"),
            ("journal", "JournalEntry"), ("je", "JournalEntry"),
        ]:
            if kw in q:
                cnt = sum(1 for _, d in graph_instance.nodes(data=True) if d.get("type") == typ)
                return False, format_response([
                    ("heading", f"Dataset Count — {typ}"),
                    ("row", ("Total records", f"{cnt:,}")),
                ]), []

    if doc_id:
        for prefix, typ in [
            ("JE_", "Journal Entry"),
            ("Bill_", "Invoice"),
            ("Order_", "Sales Order"),
        ]:
            node = f"{prefix}{doc_id}"
            if node in graph_instance:
                highlighted_nodes = [node]
                attrs = graph_instance.nodes[node]
                amt = attrs.get("amount", attrs.get("amountInCompanyCodeCurrency", "N/A"))
                return True, format_response([
                    ("heading", f"{typ} — {doc_id}"),
                    ("row", ("Amount", str(amt))),
                    ("status", "1 node highlighted in RED"),
                ]), highlighted_nodes

    return False, format_response([
        ("body", "No matching data found. Try one of these:"),
        ("divider", ""),
        ("list", [
            '"Trace order 740506"',
            '"Top products by billing"',
            '"Show broken flows"',
            '"Count all orders"',
        ]),
    ]), []


# Graph height constant — single source of truth used by both PyVis and components.html
GRAPH_HEIGHT = 640


def build_pyvis_html(
    graph: nx.DiGraph,
    highlight_nodes: Optional[List[str]] = None,
    max_nodes: int = 250,
) -> str:
    highlight_set = set(highlight_nodes or [])

    # Use exact pixel height — NOT "100%" which causes blank canvas in Streamlit
    nt = Network(
        height=f"{GRAPH_HEIGHT}px",
        width="100%",
        bgcolor="#ffffff",
        font_color="#333333",
        notebook=False,
        directed=True,
        cdn_resources="remote",
    )

    nt.set_options("""
    var options = {
      "nodes": {
        "font": {"size": 12, "face": "Arial"},
        "borderWidth": 2,
        "shadow": {"enabled": true, "color": "rgba(0,0,0,0.10)", "size": 5}
      },
      "edges": {
        "color": {"color": "#e5e7eb", "highlight": "#2563eb"},
        "width": 1.5,
        "smooth": false,
        "arrows": {"to": {"enabled": true, "scaleFactor": 0.6}}
      },
      "physics": {
        "forceAtlas2Based": {
          "gravitationalConstant": -80,
          "centralGravity": 0.02,
          "springLength": 150,
          "springConstant": 0.05
        },
        "maxVelocity": 50,
        "solver": "forceAtlas2Based",
        "timestep": 0.35,
        "stabilization": {"iterations": 150}
      },
      "interaction": {
        "hover": true,
        "tooltipDelay": 99999,
        "hideEdgesOnDrag": false,
        "navigationButtons": false
      }
    }
    """)

    subset_ids = list(graph.nodes())[:max_nodes]
    for node_id in subset_ids:
        attrs = graph.nodes[node_id]
        is_hi = node_id in highlight_set
        nt.add_node(
            node_id,
            label=str(attrs.get("label", node_id))[:20],
            title=attrs.get("title", ""),
            color="#ef4444" if is_hi else attrs.get("color", "#97c2fc"),
            size=30 if is_hi else (20 if attrs.get("type") in ["Customer", "Order"] else 14),
            borderWidth=3 if is_hi else 2,
        )

    added = set(nt.get_nodes())
    for src, tgt, data in graph.edges(data=True):
        if src in added and tgt in added:
            is_hi_e = (src in highlight_set) and (tgt in highlight_set)
            nt.add_edge(src, tgt,
                label=data.get("relationship", ""),
                color="#ef4444" if is_hi_e else "#e5e7eb",
                width=3 if is_hi_e else 1.5,
            )

    html_content = nt.generate_html(notebook=False)

    # Patch: suppress body/html margin so no whitespace appears above graph
    margin_fix = """
<style>
html, body {
    margin: 0 !important;
    padding: 0 !important;
    background: #ffffff !important;
    overflow: hidden !important;
}
#mynetwork {
    border: none !important;
}
</style>
"""
    html_content = html_content.replace("</head>", margin_fix + "\n</head>")

    # Patch: JSON click-popup handler
    popup_script = """
<style>
#o2c-popup {
    display: none;
    position: fixed;
    z-index: 99999;
    top: 16px;
    right: 16px;
    width: 264px;
    background: #ffffff;
    border-radius: 12px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.18), 0 1px 4px rgba(0,0,0,0.08);
    border: 1px solid #e5e7eb;
    font-family: 'Segoe UI', system-ui, sans-serif;
    font-size: 12px;
    overflow: hidden;
}
#o2c-popup-header {
    padding: 10px 13px 8px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    border-bottom: 1px solid #f3f4f6;
}
#o2c-popup-title { font-size: 13px; font-weight: 700; color: #111827; }
#o2c-popup-badge {
    font-size: 9px; font-weight: 700;
    padding: 2px 8px; border-radius: 20px;
    color: #fff; letter-spacing: 0.04em;
}
#o2c-popup-close {
    cursor: pointer; color: #9ca3af;
    font-size: 16px; line-height: 1;
    margin-left: 6px; user-select: none;
}
#o2c-popup-close:hover { color: #111827; }
#o2c-popup-body {
    padding: 8px 13px 4px;
    max-height: 380px;
    overflow-y: auto;
}
#o2c-popup-body::-webkit-scrollbar { width: 3px; }
#o2c-popup-body::-webkit-scrollbar-thumb { background: #e5e7eb; border-radius: 2px; }
.o2c-row {
    display: flex; gap: 8px;
    padding: 3px 0;
    border-bottom: 1px solid #f9fafb;
}
.o2c-row:last-child { border-bottom: none; }
.o2c-key {
    color: #9ca3af; min-width: 92px;
    flex-shrink: 0; font-size: 10.5px;
    padding-top: 1px;
}
.o2c-val {
    color: #111827; font-weight: 600;
    font-size: 10.5px; word-break: break-all;
}
#o2c-popup-footer {
    padding: 6px 13px 9px;
    border-top: 1px solid #f3f4f6;
    display: flex; align-items: center;
    justify-content: space-between;
}
#o2c-popup-conn { font-size: 11px; color: #2563eb; font-weight: 700; }
#o2c-popup-hint { font-size: 9.5px; color: #d1d5db; font-style: italic; }
</style>

<div id="o2c-popup">
  <div id="o2c-popup-header">
    <span id="o2c-popup-title">Node</span>
    <div style="display:flex;align-items:center;gap:5px;">
      <span id="o2c-popup-badge"></span>
      <span id="o2c-popup-close"
        onclick="document.getElementById('o2c-popup').style.display='none'">&#x2715;</span>
    </div>
  </div>
  <div id="o2c-popup-body"></div>
  <div id="o2c-popup-footer">
    <span id="o2c-popup-conn"></span>
    <span id="o2c-popup-hint">Click node to inspect</span>
  </div>
</div>

<script>
(function() {
  var TYPE_COLORS = {
    'Customer':     '#1e88e5',
    'Order':        '#ff7043',
    'LineItem':     '#66bb6a',
    'Product':      '#f59e0b',
    'Invoice':      '#ab47bc',
    'JournalEntry': '#26a69a'
  };

  function showPopup(nodeData) {
    var raw = nodeData.title;
    if (!raw) return;
    var data;
    try { data = JSON.parse(raw); } catch(e) { return; }

    document.getElementById('o2c-popup-title').textContent = data.type || 'Node';
    var badge = document.getElementById('o2c-popup-badge');
    badge.textContent = data.type || '';
    badge.style.background = TYPE_COLORS[data.type] || '#6b7280';

    var body = document.getElementById('o2c-popup-body');
    body.innerHTML = '';
    (data.rows || []).forEach(function(row) {
      var div = document.createElement('div');
      div.className = 'o2c-row';
      var k = document.createElement('span');
      k.className = 'o2c-key';
      k.textContent = row.k;
      var v = document.createElement('span');
      v.className = 'o2c-val';
      v.textContent = row.v;
      div.appendChild(k);
      div.appendChild(v);
      body.appendChild(div);
    });

    document.getElementById('o2c-popup-conn').textContent =
      'Connections: ' + (data.connections || 0);
    document.getElementById('o2c-popup').style.display = 'block';
  }

  function hidePopup() {
    document.getElementById('o2c-popup').style.display = 'none';
  }

  function attachWhenReady() {
    if (window.network && typeof window.network.on === 'function') {
      window.network.on('click', function(params) {
        if (params.nodes && params.nodes.length > 0) {
          showPopup(window.network.body.data.nodes.get(params.nodes[0]));
        } else {
          hidePopup();
        }
      });
      return;
    }
    for (var key in window) {
      try {
        if (window[key] && typeof window[key].on === 'function' &&
            window[key].body && window[key].body.data &&
            window[key].body.data.nodes) {
          var net = window[key];
          net.on('click', function(params) {
            if (params.nodes && params.nodes.length > 0) {
              showPopup(net.body.data.nodes.get(params.nodes[0]));
            } else {
              hidePopup();
            }
          });
          return;
        }
      } catch(e) {}
    }
    setTimeout(attachWhenReady, 300);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', function() {
      setTimeout(attachWhenReady, 600);
    });
  } else {
    setTimeout(attachWhenReady, 600);
  }
})();
</script>
"""
    html_content = html_content.replace("</body>", popup_script + "\n</body>")
    return html_content


def _init_session_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "<strong>DODGE AI</strong> — Ready. "
                    "Ask: <em>Trace order 740506</em>, "
                    "<em>Top products by billing</em>, or <em>Show broken flows</em>."
                ),
            }
        ]
    if "highlighted_nodes" not in st.session_state:
        st.session_state.highlighted_nodes = []
    if "user_query" not in st.session_state:
        st.session_state.user_query = ""


def main() -> None:
    st.set_page_config(page_title="Order-to-Cash Graph", layout="wide")
    _init_session_state()

    st.markdown("""
    <style>
        /* Remove ALL default Streamlit padding */
        .block-container {
            padding-top: 0.5rem !important;
            padding-bottom: 0 !important;
            padding-left: 0.75rem !important;
            padding-right: 0.75rem !important;
        }
        header[data-testid="stHeader"] {
            display: none !important;
        }
        [data-testid="stHorizontalBlock"] {
            align-items: flex-start !important;
        }
        /* Force ALL chat message text white */
        [data-testid="stChatMessage"] *,
        [data-testid="stChatMessage"] p,
        [data-testid="stChatMessage"] span,
        [data-testid="stChatMessage"] div,
        [data-testid="stChatMessage"] li,
        [data-testid="stChatMessage"] strong,
        [data-testid="stChatMessage"] em {
            color: #f8fafc !important;
        }
    </style>
    """, unsafe_allow_html=True)

    graph = load_graph()

    # Legend rendered above columns — never clipped by iframe
    legend_items = {
        "Customer":     "#1e88e5",
        "Order":        "#ff7043",
        "Line Item":    "#66bb6a",
        "Product":      "#ffee58",
        "Invoice":      "#ab47bc",
        "Journal Entry":"#26a69a",
        "Highlighted":  "#ef4444",
    }
    dots = "".join(
        f"<span style='display:inline-flex;align-items:center;gap:5px;"
        f"margin-right:14px;font-size:12px;color:#e2e8f0;'>"
        f"<span style='width:11px;height:11px;border-radius:50%;"
        f"background:{color};display:inline-block;flex-shrink:0;'></span>"
        f"{name}</span>"
        for name, color in legend_items.items()
    )
    st.markdown(
        f"<div style='background:#0f172a;padding:8px 12px;border-radius:8px;"
        f"margin-bottom:6px;display:flex;flex-wrap:wrap;align-items:center;'>"
        f"<strong style='color:#94a3b8;font-size:11px;margin-right:12px;"
        f"text-transform:uppercase;letter-spacing:0.08em;'>Legend</strong>"
        f"{dots}</div>",
        unsafe_allow_html=True,
    )

    col_graph, col_chat = st.columns([5, 1])

    with col_chat:
        st.markdown(
            "<div style='font-size:11px;font-weight:700;color:#94a3b8;"
            "text-transform:uppercase;letter-spacing:0.08em;"
            "margin-bottom:4px;'>DODGE AI</div>",
            unsafe_allow_html=True,
        )
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"], unsafe_allow_html=True)

        user_query = st.chat_input("Ask about orders, billing, products…")
        if user_query:
            st.session_state.messages.append({"role": "user", "content": user_query})
            with st.chat_message("user"):
                st.write(user_query)
            with st.chat_message("assistant"):
                with st.spinner("Processing…"):
                    has_hi, assistant_html, hi_nodes = query_graph_ai(user_query, graph)
                    st.markdown(assistant_html, unsafe_allow_html=True)
            st.session_state.highlighted_nodes = hi_nodes if has_hi else []
            st.session_state.messages.append(
                {"role": "assistant", "content": assistant_html}
            )

    with col_graph:
        html = build_pyvis_html(
            graph,
            highlight_nodes=st.session_state.get("highlighted_nodes") or [],
        )
        # height matches GRAPH_HEIGHT exactly — PyVis uses px, components.html
        # adds ~2px for iframe border, so +4 prevents a scrollbar appearing
        components.html(html, height=GRAPH_HEIGHT + 4, scrolling=False)


if __name__ == "__main__":
    main()