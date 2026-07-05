/* Dashboard bootstrap, kept in its own file so index.html carries no inline
   script and a strict `script-src 'self'` CSP applies. */
(function () {
  try { window.FUT.init(); }
  catch (e) {
    document.getElementById("app").innerHTML =
      '<p style="padding:40px;font-family:system-ui">Could not start dashboard: ' + e + "</p>";
  }
})();
