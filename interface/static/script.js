$(document).ready(function() {
  $('#form-message').submit(function(event) {
    event.preventDefault();

    var message = $('#input-message').val();
    if (message == '') return;

    var classifier = $('#select-classifier').val();
    var balanced_data = $('#checkbox-balanced').prop('checked');
    var selected_features = true; // $('#checkbox-selected').prop('checked');
    var feature_mode = $('#select-mode').val();

    $('#loading-message').show();

    $.ajax({
      url: '/recognizer',
      data: {
        message: message,
        classifier: classifier,
        balanced_data: balanced_data,
        selected_features: selected_features,
        feature_mode: feature_mode
      },
      success: function(data) {
        $('#message-intent').text(data.intent);
        $('#message-text').text(data.text);
        $('#message-details').text(data.details);
        $('#loading-message').hide();
        updateChart(data.confidence_labels, data.confidence_values);
      },
      error: function(error) {
        console.error(error);
        var show_error = confirm("Error occured, show error?");
        if (show_error) {
            alert(error.responseText);
        }
        $('#loading-message').hide();
      }
    })
  });

  updateChart = function(labels, values) {
    var myChart = new Chart(ctx, {
      type: 'horizontalBar',
      data: {
        labels: labels,
        datasets: [{
          label: 'Confidence',
          data: values,
          backgroundColor: [
              'rgba(255, 99, 132, 0.8)',
              'rgba(54, 162, 235, 0.8)',
              'rgba(255, 206, 86, 0.8)',
              'rgba(75, 192, 192, 0.8)',
              'rgba(153, 102, 255, 0.8)',
              'rgba(255, 159, 64, 0.8)'
          ],
          borderColor: [
              'rgba(255,99,132,1)',
              'rgba(54, 162, 235, 1)',
              'rgba(255, 206, 86, 1)',
              'rgba(75, 192, 192, 1)',
              'rgba(153, 102, 255, 1)',
              'rgba(255, 159, 64, 1)'
          ],
          borderWidth: 1
        }]
      },
      options: {
        scales: {
          xAxes: [{
            ticks: {
              beginAtZero:true,
            }
          }]
        }
      }
    });
  }

  var ctx = $("#chart");
  var DEFAULT_LABELS = [], DEFAULT_VALUES = [];
  updateChart(DEFAULT_LABELS, DEFAULT_VALUES);


});
