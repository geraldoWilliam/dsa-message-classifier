$(document).ready(function() {
  $('#form-message').submit(function(event) {
    event.preventDefault();

    var message = $('#input-message').val();
    if (message == '') return;

    var classifier = $('#select-classifier').val();
    var balanced_data = $('#checkbox-balanced').prop('checked');
    var selected_features = $('#checkbox-selected').prop('checked');

    $('#loading-message').show();

    $.ajax({
      url: '/recognizer',
      data: {
        message: message,
        classifier: classifier,
        balanced_data: balanced_data,
        selected_features: selected_features
      },
      success: function(data) {
        $('#message-intent').text(data.intent);
        $('#message-text').text(data.text);
        $('#message-details').text(data.details);
        $('#loading-message').hide();
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
  })
});
