import React from 'react';
import { withRouter } from 'next/router';

class Error extends React.Component {
	static getInitialProps({ req, res, pathname, query, asPath, jsonPageRes, err }) {
		let statusCode = res ? res.statusCode : null;
		let message;

		switch (Number(statusCode)) {
			case 404:
				message = 'Page Not Found';
				break;
			case 400:
				message = 'Params missing';
				break;
			case 403:
				message = 'Permission Denied';
				break;
			default:
				message = 'Something wrong!';
				break;
		}

		if (err) {
			message = err.message ? err.message : message;
			statusCode = err.statusCode;
		}

		if (res) {
			const errObj = res.locals ? res.locals.errObj : null;
			if (errObj) {
				message = errObj.errMsg ? errObj.errMsg : message;
				statusCode = errObj.statusCode;
			}
		}
		return { message, statusCode };
	}
	render() {
		return (
			<div>
				<div className="errInfo">
					<div className="backgroundDiv">
						<div>
							<div className="error">
								<div className="errCode">{this.props.statusCode}</div>
								<div className="errMessage">{this.props.message}</div>
								<br />
								{/* <a className="homeLink" href="/">
                                    <span><MDIcon size={1} path={mdiChevronLeft} /> ホームに戻す</span>
                                </a> */}
							</div>
						</div>
					</div>

					{/* <img src="/static/img/error.png" alt="Please wait..." /> */}

				</div>
				<style jsx>{`
					* {
						font-family: revert;
					}
					.error {
						// position: absolute;
                        display: flex;
                        flex-direction: column;
                        align-items: center;
                        right: 30%;
                        height: 100vh;
                        justify-content: center;
					} 
					.logo {
						flex: 1;
					}
					.backgroundDiv {
                        display: flex;
                        justify-content: flex-end;
                        position: absolute;
                        background-image: url(/static/img/error.png);
                        // filter: blur(10px);
                        background-position: left;
                        background-repeat: no-repeat;
                        background-size: 22rem;
                        height: 100%;
                        width: 620px;
					}
					.errInfo {
						height: 90vh;
						display: flex;
						justify-content: center;
						align-items: center;
					}
					.errCode {
						font-size: 10rem;
					}
					.errMessage {
						font-size: 2rem;
					}
					.homeLink {
						z-index: 5;
					}
					.homeLink > span {
						display: flex;
						font-size: 20px;
						align-items: center;
						color: #1a0be4;
						text-decoration: underline;
					}
					.backgroundImg {
						width: 10%;
						
					}
				`}</style>
			</div>
		);
	}
}

export default withRouter(Error);
