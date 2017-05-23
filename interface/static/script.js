$(document).ready(function() {
  $('#form-message').submit(function(event) {
    event.preventDefault();
    var message = $('#input-message').val();
    if (message == '') return;

    $('#loading-message').show();

    $.ajax({
      url: '/recognizer',
      data: { message: message },
      success: function(data) {
        $('#message-intent').text(data.intent);
        $('#message-text').text(data.text);
        $('#message-details').text(data.details);
        $('#loading-message').hide();
      },
      error: function(error) {
        console.error(error);
      }
    })
  })
});
