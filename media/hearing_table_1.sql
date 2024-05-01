-- phpMyAdmin SQL Dump
-- version 5.2.0
-- https://www.phpmyadmin.net/
--
-- Host: 127.0.0.1:3306
-- Generation Time: Oct 21, 2023 at 08:48 AM
-- Server version: 8.0.31
-- PHP Version: 8.0.26

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
START TRANSACTION;
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;

--
-- Database: `hearinglossdetection`
--

-- --------------------------------------------------------

--
-- Table structure for table `hearing_table`
--

DROP TABLE IF EXISTS `hearing_table`;
CREATE TABLE IF NOT EXISTS `hearing_table` (
  `user_id` int NOT NULL AUTO_INCREMENT,
  `user_dates` date DEFAULT NULL,
  `user_username` longtext,
  `user_email` longtext,
  `user_password` longtext,
  `user_contact` longtext,
  `user_file` varchar(100) DEFAULT NULL,
  `user_status` longtext,
  `otp_status` longtext NOT NULL,
  `otp` int DEFAULT NULL,
  `Last_Login_Date` date DEFAULT NULL,
  `Last_Login_Time` time(6) DEFAULT NULL,
  `No_Of_Times_Login` int DEFAULT NULL,
  PRIMARY KEY (`user_id`)
) ENGINE=MyISAM AUTO_INCREMENT=4 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

--
-- Dumping data for table `hearing_table`
--

INSERT INTO `hearing_table` (`user_id`, `user_dates`, `user_username`, `user_email`, `user_password`, `user_contact`, `user_file`, `user_status`, `otp_status`, `otp`, `Last_Login_Date`, `Last_Login_Time`, `No_Of_Times_Login`) VALUES
(3, '2023-10-21', 'user_-11', 'user@gmail.com', '111', '5555555555', 'images/2022-11-14.png', 'Accepted', 'verified', 4135, '2023-10-21', '14:14:41.000000', 1);
COMMIT;

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
