import React, { Fragment, FunctionComponent, useState, useEffect, useRef } from 'react';
import clsx from 'clsx';
import theme from 'compDir/Theme/tasktracker';
import { createStyles, makeStyles, ThemeProvider } from '@material-ui/core/styles';
import commonCss from 'compDir/Theme/commonCss';
import MDIcon from '@mdi/react';
import ListAltIcon from '@material-ui/icons/ListAlt';
import AssignmentTurnedInIcon from '@material-ui/icons/AssignmentTurnedIn';
import {
	mdiChevronLeft, mdiShape, mdiMenu, mdiCheckboxMarkedCircleOutline, mdiCalendarRange,
	mdiYoutubeSubscription, mdiDownloadCircleOutline, mdiStopCircleOutline, mdiAutorenew, mdiDelete, mdiChartBar, mdiHelpBox, mdiForum
} from '@mdi/js';
import { initSidebar } from 'compDir/SideBar';
import ActiveLink from 'compDir/ActiveLink';
import ColorButton from 'compDir/Button';
import {
	Box, Grid, List, ListItem, ListItemText,
	Divider, Avatar, Typography, Tooltip, Paper, TableContainer, Table, TableHead, TableCell, TableRow, TableBody, Link
} from '@material-ui/core';
import { ICAUser, ICsvReport } from 'interfaceDir';
import { Utils } from 'servDir';
import { DownloadStatus, UserRole, TIME_ZONE } from 'constDir';
import Modal from 'compDir/Modal';
import LinearProgress from 'compDir/LinearProgress';
import moment from 'moment-timezone';
import { toast } from 'material-react-toastify';

moment.tz.setDefault(TIME_ZONE);

export interface IStateToProps {
	openSideBar: boolean
}

export interface IDispatchToProps {
	toggleSideBar(): any
	(): any
	cancelDownload(csvId): any
	removeCsvFile(csvId): any
	saveUserInfo(userInfo): Promise<any>
}

interface IProps extends IDispatchToProps, IStateToProps {
}

const useStyles = makeStyles(() =>
	createStyles({
		...commonCss,
		root: {
			height: '100vh',
			// width: '100%',
			// position: 'fixed',
			// overflowY: 'auto'
		},
		hide: {
			display: 'none'
		},
		leftNavItem: {
			height: theme.spacing(6),

			['&:hover']: {
				backgroundColor: theme.palette.secondary.light
			},
			'&.active': {
				backgroundColor: theme.palette.primary.main
			}
		},
		menu: {
			padding: 0,
			marginBottom: 20
		},
		leftNavButton: {
			padding: 8,
			['&:hover']: {
				backgroundColor: '#f57f1733'
			}
		},
		procMasterIcon: {
			width: theme.spacing(3),
			height: theme.spacing(3),
			color: theme.palette.primary.contrastText
		},
		avatarIcon: {
			width: theme.spacing(5),
			height: theme.spacing(5),
			backgroundColor: theme.palette.secondary.light
		},
		settingTitle: {
			color: theme.palette.secondary.main
		},
		content: {
			flex: 1,
			padding: 20,
			display: 'flex',
			flexDirection: 'column',
			overflow: 'auto',
			backgroundColor: '#FAFAFA'
		},
		modal: {
			margin: 0,
			position: 'absolute',
			top: 0
		},
	})
);

const SideBar = initSidebar(theme.spacing(30), theme.spacing(7.5));

const MainLayout: FunctionComponent<IProps> = (props) => {
	const classes = useStyles();
	const [dlModal, setDlModal] = useState(false);
	const [userInfoModal, setUserInfoModal] = useState(false);
	const userInfoRef = useRef<HTMLInputElement>(null);

	const userInfo = props.userInfo;
	const openSideBar = props.openSideBar;

	useEffect(() => {
	}, []);

	const updateUserInfo = () => {
		props.saveUserInfo(userInfoRef.current?.value)
			.then(() => {
				toast.success('ユーザー情報の編集が成功しました');
				setUserInfoModal(false);
			})
			.catch((e) => {
				toast.error('ユーザー情報の編集が失敗しました.');
				console.error(e);
			});
	};

	return (
		<ThemeProvider theme={theme}>
			<Grid container className={classes.root}>
				<SideBar isFullSidebar={openSideBar}>
					<List className={classes.menu}>
						<ListItem disableGutters className={openSideBar ? classes.titleLeftNavOpen : classes.titleLeftNavClose}>
							<Grid container alignItems='center' justify={openSideBar ? 'space-around' : 'center'} wrap='nowrap'>
								{
									openSideBar && (
										<Grid item style={{ position: 'relative' }}>
											{/* <Box fontSize={24} fontWeight='fontWeightBold'>TaskTracker</Box> */}
											<img style={{ width: '12rem' }} src="/static/img/logo/adstracker.png" />
											{/* {
												process.env.NEXT_PUBLIC_NODE_ENV != 'production' && (
													<Box component={'span'} position='absolute' marginX={1} style={{ backgroundColor: 'blue' }}
														right={0} bottom={0} fontSize={8} fontWeight='fontWeightLight'>
														{process.env.NEXT_PUBLIC_NODE_ENV}
													</Box>
												)
											} */}
										</Grid>
									)
								}
								<Grid item>
									<ColorButton btnType='icon' onClick={props.toggleSideBar} className={clsx(classes.leftNavButton)}>
										<MDIcon size={'24px'} path={openSideBar ? mdiChevronLeft : mdiMenu} color={theme.palette.primary.main} />
									</ColorButton>
								</Grid>
							</Grid>
						</ListItem>
						<Divider />
						<ListItem disableGutters>
							<Grid container alignItems='center' justify={openSideBar ? 'flex-start' : 'center'} wrap='nowrap'>
								<Grid item>
									<Box paddingX={1}>
										<Avatar className={classes.avatarIcon} color={theme.palette.secondary.light}></Avatar>
									</Box>
								</Grid>
								<Grid item>
									<Box paddingX={1} display={openSideBar ? 'block' : 'none'} >
										<Box component="div" overflow='hidden' whiteSpace="nowrap" textOverflow="ellipsis" fontWeight='fontWeightRegular'>
											{userInfo.name}
										</Box>
									</Box>
								</Grid>
							</Grid>
						</ListItem>
						<Divider style={{ marginBottom: 8 }} />
						<ActiveLink href="/" activeClassName="active">
							<Tooltip title='Dashboard' placement="right" disableHoverListener={openSideBar} arrow>
								<ListItem className={classes.leftNavItem} button>
									<Grid container alignItems='center' justify={openSideBar ? 'flex-start' : 'center'} wrap='nowrap' spacing={1}>
										<Grid item style={{ display: 'flex' }}>
											<MDIcon color={theme.palette.background.default} size={'24px'} path={mdiCheckboxMarkedCircleOutline} />
										</Grid>
										{
											openSideBar && (
												<Grid item>
													<Typography component='div' variant="body1">
														<Box fontWeight='fontWeightRegular'>Dashboard</Box>
													</Typography>
												</Grid>
											)
										}
									</Grid>
								</ListItem>
							</Tooltip>
						</ActiveLink>
						<ActiveLink href="/statistic" activeClassName="active">
							<Tooltip title='Statistic' placement="right" disableHoverListener={openSideBar} arrow>
								<ListItem className={classes.leftNavItem} button>
									<Grid container alignItems='center' justify={openSideBar ? 'flex-start' : 'center'} wrap='nowrap' spacing={1}>
										<Grid item style={{ display: 'flex' }}>
											<MDIcon color={theme.palette.background.default} size={'24px'} path={mdiChartBar} />
										</Grid>
										{
											openSideBar && (
												<Grid item>
													<Typography component='div' variant="body1">
														<Box fontWeight='fontWeightRegular'>Statistic</Box>
													</Typography>
												</Grid>
											)
										}
									</Grid>
								</ListItem>
							</Tooltip>
						</ActiveLink>
						
					</List>
					<CopyrightComponent />
				</SideBar>
				
				<Grid item className={classes.content}>
					{props.children}
				</Grid>
			</Grid>
		</ThemeProvider >
	);
};

const CopyrightComponent: FC<IProps> = (props: IProps) => (
	<Box color='white' component={'p'} position='fixed' marginX={'10px'} marginY={'5px'}
		left={0} bottom={0} fontSize={8} fontWeight='fontWeightLight'>
		©{new Date().getFullYear()} MSE11-HCM All Rights Reserved. ({process.env.UI_VERSION})
	</Box>
);

export default MainLayout;
