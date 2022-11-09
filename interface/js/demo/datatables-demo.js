// Call the dataTables jQuery plugin
$(document).ready(function() {
  $('#dataTable').DataTable({
    order: [[1, 'desc']],
});
});
$(document).ready(function() {
  $('#dataTable2').DataTable({
    order: [[2, 'desc']],
});
});

$(document).ready(function() {
  $('#dataTable3').DataTable({
    order: [[4, 'asc']],
});
});