
// loads the fleet data from the json files and shows the table and the chart

var fleet = [];     
var summary = {};   


var splitFilter = "all";
var statusFilter = "all";
var searchText = "";
var sortColumn = "rul_pred_last";
var sortAscending = true;


var columns = [
  ["engine", "Engine"],
  ["split", "Split"],
  ["cycles", "Cycles"],
  ["rul_true_last", "True RUL"],
  ["rul_pred_last", "Predicted RUL"],
  ["error", "Error"],
  ["anom_last", "Anomaly"],
  ["status", "Status"]
];

var splitNames = { val: "Validation", test: "Test" };
var statusNames = { critical: "Critical", watch: "Watch", healthy: "Healthy" };


// less than 30 cycles left = critical, less than 60 = watch
function getStatus(rul) {
  if (rul < 30) {
    return "critical";
  }
  if (rul < 60) {
    return "watch";
  }
  return "healthy";
}


function loadData() {
  fetch("data/fleet.json")
    .then(function (res) {
      return res.json();
    })
    .then(function (data) {
      fleet = data.fleet;

      // add the error and the status to every engine
      for (var i = 0; i < fleet.length; i++) {
        fleet[i].error = fleet[i].rul_pred_last - fleet[i].rul_true_last;
        fleet[i].status = getStatus(fleet[i].rul_pred_last);
      }

      return fetch("data/model_summary.json");
    })
    .then(function (res) {
      return res.json();
    })
    .then(function (data) {
      summary = data;
      showSummary();
      showHeader();
    
      showTable();
      showChart();
      showResidualChart();
      showUncertaintyChart();
    })
    
    .catch(function (err) {
      var box = document.getElementById("error");
      box.hidden = false;
      box.textContent = "Could not load the data (" + err.message + "). Try running python -m http.server 8000 inside the web folder.";
    });
}


function makeBox(label, value, hint) {
  return '<div class="kpi">' +
    '<div class="label">' + label + '</div>' +
    '<div class="value">' + value + '</div>' +
    '<div class="hint">' + hint + '</div>' +
    '</div>';
}


function showSummary() {
  
  var criticalCount = 0;
  for (var i = 0; i < fleet.length; i++) {
    if (fleet[i].status == "critical") {
      criticalCount = criticalCount + 1;
    }
  }

  var html = "";
  html += makeBox("Validation RMSE", summary.val_rmse.toFixed(1), "cycles");
  html += makeBox("Validation MAE", summary.val_mae.toFixed(1), "cycles");
  html += makeBox("Engines", fleet.length, summary.n_val + " validation, " + summary.n_test + " test");
  html += makeBox("Critical now", criticalCount, "predicted RUL under 30");
  document.getElementById("kpis").innerHTML = html;
}


function showHeader() {
  var html = "";
  for (var i = 0; i < columns.length; i++) {
    html += '<th><button type="button" onclick="sortBy(\'' + columns[i][0] + '\')">' + columns[i][1] + '</button></th>';
  }
  document.getElementById("head").innerHTML = html;
}


// called when a column title is clicked
function sortBy(column) {
  if (sortColumn == column) {
    sortAscending = !sortAscending;   
  } else {
    sortColumn = column;
    sortAscending = true;
  }
  showTable();
}


// goes through all engines and keeps the ones that match the filters
function getRows() {
  var rows = [];
  for (var i = 0; i < fleet.length; i++) {
    var e = fleet[i];
    if (splitFilter != "all" && e.split != splitFilter) {
      continue;
    }
    if (statusFilter != "all" && e.status != statusFilter) {
      continue;
    }
    if (searchText != "" && String(e.engine).indexOf(searchText) != 0) {
      continue;
    }
    rows.push(e);
  }

  rows.sort(function (a, b) {
    var x = a[sortColumn];
    var y = b[sortColumn];
    var result;
    if (typeof x == "string") {
      result = x.localeCompare(y);
    } else {
      result = x - y;
    }
    if (!sortAscending) {
      result = -result;
    }
    return result;
  });

  return rows;
}


function showTable() {
  var rows = getRows();
  var html = "";

  for (var i = 0; i < rows.length; i++) {
    var e = rows[i];
    var error = e.error.toFixed(1);
    if (e.error > 0) {
      error = "+" + error;
    }

    html += "<tr>";
    html += '<td><a href="engine.html?key=' + e.key + '">#' + e.engine + "</a></td>";
    html += "<td>" + splitNames[e.split] + "</td>";
    html += "<td>" + e.cycles + "</td>";
    html += "<td>" + e.rul_true_last.toFixed(1) + "</td>";
    html += "<td>" + e.rul_pred_last.toFixed(1) + " +/- " + (1.96 * e.std_last).toFixed(1) + "</td>";
    html += "<td>" + error + "</td>";
    html += "<td>" + e.anom_last.toFixed(1) + "x</td>";
    html += '<td class="b-' + e.status + '">' + statusNames[e.status] + "</td>";
    html += "</tr>";
  }

  if (rows.length == 0) {
    html = '<tr><td colspan="8">No engines match the filters.</td></tr>';
  }

  document.getElementById("rows").innerHTML = html;
  document.getElementById("count").textContent = rows.length + " of " + fleet.length + " engines";
}


// makes the list of x and y values forone split (val or test)
function makeTrace(split, color) {
  var xs = [];
  var ys = [];
  var names = [];
  for (var i = 0; i < fleet.length; i++) {
    if (fleet[i].split == split) {
      xs.push(fleet[i].rul_true_last);
      ys.push(fleet[i].rul_pred_last);
      names.push("Engine #" + fleet[i].engine);
    }
  }
  return {
    x: xs,
    y: ys,
    text: names,
    name: splitNames[split],
    mode: "markers",
    type: "scatter",
    marker: { color: color, size: 8 }
  };
}


function showChart() {
  var valTrace = makeTrace("val", "blue");
  var testTrace = makeTrace("test", "orange");


  var lineTrace = {
    x: [0, 130],
    y: [0, 130],
    name: "Perfect",
    mode: "lines",
    line: { color: "gray", dash: "dash" }
  };

  var layout = {
    xaxis: { title: "True RUL (cycles)" },
    yaxis: { title: "Predicted RUL (cycles)" },
    margin: { t: 20 }
  };

  Plotly.newPlot("scatter", [valTrace, testTrace, lineTrace], layout);
}

e
function showResidualChart() {
  var errors = [];
  for (var i = 0; i < fleet.length; i++) {
    errors.push(fleet[i].error);
  }

  var trace = {
    x: errors,
    type: "histogram",
    nbinsx: 30,
    marker: { color: "steelblue" },
    name: "Engines"
  };

  var layout = {
    xaxis: { title: "Prediction error (cycles)" },
    yaxis: { title: "Number of engines" },
    margin: { t: 20 },
    
    shapes: [{
      type: "line", x0: 0, x1: 0, y0: 0, y1: 1, yref: "paper",
      line: { color: "red", dash: "dash" }
    }]
  };

  Plotly.newPlot("residual", [trace], layout);
}


// uncertainty (95% interval) on the x axis, size of the error on the y axis
function makeErrorTrace(split, color) {
  var xs = [];
  var ys = [];
  var names = [];
  for (var i = 0; i < fleet.length; i++) {
    if (fleet[i].split == split) {
      xs.push(1.96 * fleet[i].std_last);
      ys.push(Math.abs(fleet[i].error));
      names.push("Engine #" + fleet[i].engine);
    }
  }
  return {
    x: xs,
    y: ys,
    text: names,
    name: splitNames[split],
    mode: "markers",
    type: "scatter",
    marker: { color: color, size: 8 }
  };
}


function showUncertaintyChart() {
  var valTrace = makeErrorTrace("val", "blue");
  var testTrace = makeErrorTrace("test", "orange");

  
  var biggest = 0;
  for (var i = 0; i < fleet.length; i++) {
    var width = 1.96 * fleet[i].std_last;
    var err = Math.abs(fleet[i].error);
    if (width > biggest) { biggest = width; }
    if (err > biggest) { biggest = err; }
  }

  // where error = interval width
  var lineTrace = {
    x: [0, biggest],
    y: [0, biggest],
    name: "Error = interval",
    mode: "lines",
    line: { color: "gray", dash: "dash" }
  };

  var layout = {
    xaxis: { title: "Model uncertainty: 95% interval half-width (cycles)" },
    yaxis: { title: "Absolute error (cycles)" },
    margin: { t: 20 }
  };

  Plotly.newPlot("uncertainty", [valTrace, testTrace, lineTrace], layout);
}



document.getElementById("fSplit").onchange = function () {
  splitFilter = this.value;
  showTable();
};

document.getElementById("fStatus").onchange = function () {
  statusFilter = this.value;
  showTable();
};

document.getElementById("fSearch").oninput = function () {
  searchText = this.value;
  showTable();
};

loadData();