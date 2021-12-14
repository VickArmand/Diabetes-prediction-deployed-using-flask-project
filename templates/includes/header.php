<!DOCTYPE html>
<html>
<head>
	<title>Dashboard</title>
	<meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">

    <!-- Bootstrap CSS -->
    <link rel="stylesheet" href="includes/css/bootstrap.css">
    
</head>
<body style="background: #CCC;">
<span class="header">
    <nav class="navbar navbar-expand-sm navbar-dark bg-dark ">
      <div class="container">
      <a href="#" class="navbar-brand"><h3>Hostel Management System</h3></a>

  <ul class="nav navbar-nav navbar-right">
   

    <?php
     session_start();
   
    if (isset($_SESSION['users'])||isset($_SESSION['admin']))
    {
?>
<li><a href="users/logout.php" type="button" class="btn btn-outline-light ml-3">Log Out</a></li>
<?php
    }
    else{
?>
</ul>
<ul class="nav navbar-nav navbar-right">
<li><a href="users/login.php"type="button" class="btn btn-outline-light ml-3"><i class="fa fa-users"></i> User Login</a></li>
<li><a href="admin/login.php"type="button" class="btn btn-outline-light ml-3"><i class="fa fa-user"></i> Admin Login</a></li>
	<style>
  	ul,li{
      padding:5px;
    }	
	</style>
</ul>
<?php    
}
  ?>
  </ul>
  
 
    </div>

  </nav>
  </span>
    <script src="js/bootstrap.js"></script>
  </body>
</html>